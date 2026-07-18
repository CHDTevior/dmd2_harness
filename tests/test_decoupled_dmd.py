import unittest

import torch

from src.dmd2_firered.decoupled_dmd import (
    coupled_raw_gradient,
    decoupled_raw_gradient,
    noised_latents_from_noise,
    normalize_and_clip_gradient,
    sample_constrained_ca_time,
)


class DecoupledDMDTest(unittest.TestCase):
    def test_forced_tie_matches_coupled_raw_and_normalized_gradient(self) -> None:
        generator = torch.Generator().manual_seed(17)
        fake_x0 = torch.randn((2, 4, 8, 8), generator=generator, dtype=torch.float32)
        real_cond_x0 = torch.randn((2, 4, 8, 8), generator=generator, dtype=torch.float32)
        real_uncond_x0 = torch.randn((2, 4, 8, 8), generator=generator, dtype=torch.float32)

        coupled = coupled_raw_gradient(
            fake_x0=fake_x0,
            real_cond_x0=real_cond_x0,
            real_uncond_x0=real_uncond_x0,
            guidance_scale=4.0,
        )
        decoupled, _, _ = decoupled_raw_gradient(
            fake_dm_x0=fake_x0,
            real_cond_dm_x0=real_cond_x0,
            real_cond_ca_x0=real_cond_x0,
            real_uncond_ca_x0=real_uncond_x0,
            guidance_scale=4.0,
        )
        torch.testing.assert_close(decoupled, coupled, rtol=1.0e-6, atol=1.0e-6)

        coupled_normalized, coupled_norm = normalize_and_clip_gradient(coupled, eps=1.0e-6, clip=10.0)
        decoupled_normalized, decoupled_norm = normalize_and_clip_gradient(
            decoupled, eps=1.0e-6, clip=10.0
        )
        torch.testing.assert_close(decoupled_norm, coupled_norm, rtol=0.0, atol=0.0)
        torch.testing.assert_close(decoupled_normalized, coupled_normalized, rtol=1.0e-6, atol=1.0e-6)

    def test_forced_time_and_noise_one_batch_integration(self) -> None:
        generator = torch.Generator().manual_seed(29)
        student_x0 = torch.randn((2, 4, 6, 6), generator=generator, dtype=torch.float32)
        forced_t = torch.tensor([0.23, 0.81], dtype=torch.float32)
        forced_noise = torch.randn(student_x0.shape, generator=generator, dtype=torch.float32)
        x_t = noised_latents_from_noise(student_x0, forced_t, forced_noise)
        fake_v = torch.randn(student_x0.shape, generator=generator, dtype=torch.float32)
        real_cond_v = torch.randn(student_x0.shape, generator=generator, dtype=torch.float32)
        real_uncond_v = torch.randn(student_x0.shape, generator=generator, dtype=torch.float32)
        t_expanded = forced_t.view(2, 1, 1, 1)
        fake_x0 = x_t - t_expanded * fake_v
        real_cond_x0 = x_t - t_expanded * real_cond_v
        real_uncond_x0 = x_t - t_expanded * real_uncond_v

        legacy = coupled_raw_gradient(
            fake_x0=fake_x0,
            real_cond_x0=real_cond_x0,
            real_uncond_x0=real_uncond_x0,
            guidance_scale=4.0,
        )
        treatment, delta_dm, delta_ca = decoupled_raw_gradient(
            fake_dm_x0=fake_x0,
            real_cond_dm_x0=real_cond_x0,
            real_cond_ca_x0=real_cond_x0,
            real_uncond_ca_x0=real_uncond_x0,
            guidance_scale=4.0,
        )
        self.assertTrue(torch.isfinite(delta_dm).all().item())
        self.assertTrue(torch.isfinite(delta_ca).all().item())
        torch.testing.assert_close(treatment, legacy, rtol=1.0e-6, atol=1.0e-6)

    def test_focused_ca_schedule_respects_all_four_student_stages(self) -> None:
        stage_times = (1.0, 0.75, 0.5, 0.25)
        for stage_idx, t_gen_value in enumerate(stage_times):
            t_gen = torch.full((4096,), t_gen_value, dtype=torch.float32)
            samples = sample_constrained_ca_time(
                t_gen=t_gen,
                low=0.02,
                high=0.98,
                generator=torch.Generator().manual_seed(100 + stage_idx),
            )
            upper = min(0.98, t_gen_value)
            self.assertGreaterEqual(float(samples.min()), 0.02)
            self.assertLess(float(samples.max()), upper)

    def test_focused_ca_rejects_empty_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty interval"):
            sample_constrained_ca_time(
                t_gen=torch.tensor([0.02], dtype=torch.float32),
                low=0.02,
                high=0.98,
            )


if __name__ == "__main__":
    unittest.main()
