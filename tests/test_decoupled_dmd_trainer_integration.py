import os
import unittest

os.environ.setdefault("FIRERED_DISABLE_FLASH_ATTN", "1")
os.environ.setdefault("TWINFLOW_SRC", "/vepfs-cnbja62d5d769987/suntengjiao/TwinFlow/src")

import torch
from torch import nn

from scripts.train_firered_dmd2_full_fsdp import DMD2FullOfficialMethod
from src.dmd2_firered.decoupled_dmd import DMGradientOverrides


class TinyVelocityModel(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.calls = 0

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition) -> torch.Tensor:
        self.calls += 1
        prompt = condition[0].float().mean(dim=tuple(range(1, condition[0].dim())))
        prompt = prompt.view(prompt.shape[0], *([1] * (x_t.dim() - 1)))
        return self.scale * x_t.float() + 0.05 * prompt


class FrozenTeacher:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor, condition) -> torch.Tensor:
        self.calls += 1
        prompt = condition[0].float().mean(dim=tuple(range(1, condition[0].dim())))
        prompt = prompt.view(prompt.shape[0], *([1] * (x_t.dim() - 1)))
        return 0.3 * x_t.float() + 0.1 * prompt


class DecoupledDMDTrainerIntegrationTest(unittest.TestCase):
    def test_single_batch_backward_uses_decoupled_teacher_paths(self) -> None:
        torch.manual_seed(123)
        method = DMD2FullOfficialMethod(
            {
                "critic_mode": "separate_full",
                "decoupled_dmd": True,
                "decoupled_ca_mode": "constrained",
                "ca_guidance_scale": 4.0,
                "dm_noise_t_min": 0.02,
                "dm_noise_t_max": 0.98,
                "ca_noise_t_min": 0.02,
                "ca_noise_t_max": 0.98,
                "teacher_match_loss_weight": 0.0,
                "dm_loss_weight": 1.0,
                "fake_loss_weight": 1.0,
                "gan_loss_weight": 0.0,
                "gan_classifier_loss_weight": 0.0,
                "real_guidance_scale": 4.0,
                "fake_guidance_scale": 1.0,
                "student_train_sampling_steps": 4,
                "student_train_backprop_mode": "single_step",
                "dfake_gen_update_ratio": 5,
                "dm_grad_clip": 10.0,
                "dm_grad_eps": 1.0e-6,
            }
        )
        student = TinyVelocityModel(0.2)
        fake = TinyVelocityModel(0.25)
        teacher = FrozenTeacher()
        target = torch.randn((2, 4, 4, 4), dtype=torch.float32)
        source = torch.randn_like(target)
        prompt = torch.ones((2, 3, 5), dtype=torch.float32)
        uncond = torch.zeros_like(prompt)
        prompt_mask = torch.ones((2, 3), dtype=torch.long)

        loss, metrics = method.training_step_backward(
            student_model_fn=student,
            fake_model_fn=fake,
            teacher_model_fn=teacher,
            gan_classifier_gen_fn=None,
            gan_classifier_train_fn=None,
            target_latents=target,
            source_latents=source,
            prompt_embeds=prompt,
            prompt_mask=prompt_mask,
            uncond_embeds=uncond,
            uncond_mask=prompt_mask,
            dtype=torch.float32,
            backward_scale=1.0,
            compute_student_gradient=True,
        )

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(teacher.calls, 3)
        self.assertEqual(fake.calls, 2)
        self.assertGreater(student.calls, 0)
        self.assertEqual(float(metrics["ca_constraint_violation"]), 0.0)
        self.assertGreaterEqual(float(metrics["t_ca_mean"]), 0.02)
        self.assertLess(float(metrics["t_ca_mean"]), float(metrics["t_gen_mean"])+1.0e-7)
        self.assertGreater(float(metrics["delta_dm_norm"]), 0.0)
        self.assertGreater(float(metrics["delta_ca_norm"]), 0.0)
        self.assertIsNotNone(student.scale.grad)
        self.assertIsNotNone(fake.scale.grad)
        self.assertTrue(torch.isfinite(student.scale.grad).item())
        self.assertTrue(torch.isfinite(fake.scale.grad).item())

    def test_forced_tie_matches_coupled_training_step_loss_dm(self) -> None:
        base_cfg = {
            "critic_mode": "separate_full",
            "teacher_match_loss_weight": 0.0,
            "dm_loss_weight": 1.0,
            "fake_loss_weight": 1.0,
            "gan_loss_weight": 0.0,
            "gan_classifier_loss_weight": 0.0,
            "real_guidance_scale": 4.0,
            "fake_guidance_scale": 1.0,
            "student_train_sampling_steps": 1,
            "student_train_backprop_mode": "single_step",
            "dfake_gen_update_ratio": 5,
            "dm_grad_clip": 10.0,
            "dm_grad_eps": 1.0e-6,
        }
        coupled = DMD2FullOfficialMethod({**base_cfg, "decoupled_dmd": False})
        decoupled = DMD2FullOfficialMethod(
            {
                **base_cfg,
                "decoupled_dmd": True,
                "decoupled_ca_mode": "full",
                "ca_guidance_scale": 4.0,
                "dm_noise_t_min": 0.02,
                "dm_noise_t_max": 0.98,
                "ca_noise_t_min": 0.02,
                "ca_noise_t_max": 0.98,
            }
        )
        draw_generator = torch.Generator().manual_seed(919)
        target = torch.randn((2, 4, 4, 4), generator=draw_generator, dtype=torch.float32)
        source = torch.randn(target.shape, generator=draw_generator, dtype=torch.float32)
        prompt = torch.randn((2, 3, 5), generator=draw_generator, dtype=torch.float32)
        uncond = torch.randn((2, 3, 5), generator=draw_generator, dtype=torch.float32)
        prompt_mask = torch.ones((2, 3), dtype=torch.long)
        tied_t = torch.tensor([0.31, 0.77], dtype=torch.float32)
        tied_noise = torch.randn(target.shape, generator=draw_generator, dtype=torch.float32)
        overrides = DMGradientOverrides(
            t_dm=tied_t,
            dm_noise=tied_noise,
            t_ca=tied_t,
            ca_noise=tied_noise,
        )

        def run(method: DMD2FullOfficialMethod):
            student = TinyVelocityModel(0.2)
            fake = TinyVelocityModel(0.25)
            teacher = FrozenTeacher()
            torch.manual_seed(2026)
            _, metrics = method.training_step_backward(
                student_model_fn=student,
                fake_model_fn=fake,
                teacher_model_fn=teacher,
                gan_classifier_gen_fn=None,
                gan_classifier_train_fn=None,
                target_latents=target,
                source_latents=source,
                prompt_embeds=prompt,
                prompt_mask=prompt_mask,
                uncond_embeds=uncond,
                uncond_mask=prompt_mask,
                dtype=torch.float32,
                backward_scale=1.0,
                compute_student_gradient=True,
                dm_gradient_overrides=overrides,
            )
            return metrics["loss_dm"], student.scale.grad

        coupled_loss_dm, coupled_grad = run(coupled)
        decoupled_loss_dm, decoupled_grad = run(decoupled)
        torch.testing.assert_close(decoupled_loss_dm, coupled_loss_dm, rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(decoupled_grad, coupled_grad, rtol=1.0e-5, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
