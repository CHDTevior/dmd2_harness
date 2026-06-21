# DMD2 Method Notes

Source repo: `/vepfs-cnbja62d5d769987/suntengjiao/distill/dmd2`

Important files:

- `main/train_sd.py`: training loop and optimizer schedule.
- `main/sd_unified_model.py`: wraps student generator and guidance model.
- `main/sd_guidance.py`: real/fake score logic and GAN classifier loss.
- `main/sd_image_dataset.py`: real latent LMDB dataset.
- `main/train_sd_ode.py`: ODE regression pretraining used by 1-step SDXL.
- `experiments/sdxl/*.sh`: reference hyperparameters.

## Components

| Component | Upstream name | Trainable | Role |
| --- | --- | --- | --- |
| Student | `feedforward_model` | yes | Few-step generator being distilled |
| Real teacher | `real_unet` | no | Frozen score model for target distribution |
| Fake critic | `fake_unet` | yes | Learns score of current generated distribution |
| GAN head | `cls_pred_branch` | yes | Distinguishes real latents from generated latents |
| Text encoder / VAE | `text_encoder`, `vae` | no | Conditioning and latent/image conversion |

## Generator Update

In `SDUniModel.forward(... generator_turn=True)`:

1. Sample noise or a noisy denoising input.
2. Student predicts noise at a conditioning timestep.
3. Convert predicted noise to `x0` using `get_x0_from_noise`.
4. Pass generated latent to `SDGuidance(generator_turn=True)`.
5. `SDGuidance.compute_distribution_matching_loss`:
   - adds random noise to the generated latent at random timestep `t`;
   - predicts denoised image with fake critic;
   - predicts denoised image with frozen real teacher;
   - computes normalized score-difference gradient;
   - applies a detached MSE surrogate so gradients flow to the student.
6. Optional generator GAN loss pushes generated latents to look real to the classifier head.

The upstream formula in code is:

```text
p_real = latents - pred_real_image
p_fake = latents - pred_fake_image
grad = (p_real - p_fake) / mean(abs(p_real))
loss_dm = 0.5 * mse(latents, stopgrad(latents - grad))
```

## Guidance Update

In `SDGuidance(guidance_turn=True)`:

1. `compute_loss_fake` trains fake critic to denoise current generated latents.
2. If `cls_on_clean_image` is enabled, classifier loss compares:
   - real VAE latents from `SDImageDatasetLMDB`;
   - generated latents from the student.

The default SDXL 4-step run updates generator every `dfake_gen_update_ratio=5` steps and updates guidance every step.

## Multi-Step DMD2

The upstream 4-step SDXL recipe uses:

- `--denoising`
- `--num_denoising_step 4`
- `--denoising_timestep 1000`
- `--backward_simulation`

This matters because pure one-shot training on initial noise does not match inference-time inputs for later steps. Backward simulation generates intermediate latents during training so the student sees the same type of inputs it will see during 4-step inference.

## 1-Step DMD2

Upstream SDXL 1-step is marked work-in-progress and requires regression/ODE pretraining:

1. Generate or download 10K noise-image ODE pairs.
2. Pretrain the student with regression loss.
3. Run DMD2 training from that checkpoint.

For FireRed, we should not start with 1-step full DMD2 until 4-step LoRA DMD2 is stable.

