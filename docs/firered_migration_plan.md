# FireRed Gray DMD2 Migration Plan

## Current FireRed Inputs

| Item | Path / field |
| --- | --- |
| Base model | `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/ckpts` |
| Original gray LoRA | `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/ckpts/adapter_gray/adapter` |
| Merged full model | `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/ckpts_gray_lora_merged_v1` |
| Train JSONL | `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/clay_meta/clay/clay.jsonl` |
| Local COS mirror | `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/cos_data_clay` |
| Source image field | `source_image` |
| Target image field | `edit_image` |
| Conditional embedding | `embeddings_tensor_en` |
| Unconditional embedding | `embeddings_tensor_droptext` |

## What Must Change From SDXL DMD2

| SDXL DMD2 | FireRed replacement |
| --- | --- |
| `UNet2DConditionModel` | QwenImageEdit transformer wrapper |
| SDXL text tokenizers | existing offline VLM embeddings from JSONL |
| SDXL `added_cond_kwargs` | FireRed source image latent + prompt mask/embeds |
| SD VAE latent shape `[B,4,H/8,W/8]` | Qwen/ImageEdit latent packing used by FireRed pipeline |
| `get_x0_from_noise` epsilon formula | FireRed/Qwen scheduler or flow-to-x0 conversion |
| Conv classifier on UNet bottleneck | classifier head on Qwen transformer hidden states or latent patches |
| LMDB of real SDXL VAE latents | local precomputed target gray latents with source/prompt conditions |
| `real_guidance_scale=8` | `CFG=0` for FireRed gray eval and likely training |

## Proposed LoRA DMD2 Design

Start with three logical adapters:

| Adapter | Role | Trainable |
| --- | --- | --- |
| `teacher_gray` | original FireRed gray LoRA capability | no |
| `student_dmd2` | few-step DMD2 student | yes |
| `fake_critic` | generated-distribution score estimator | yes |

Implementation options:

1. One frozen base model with adapter switching for teacher/student/fake critic. This is memory efficient but requires careful no-grad boundaries and adapter isolation.
2. Two model copies: one frozen teacher, one base with student/fake adapters. This is safer for engineering and still much cheaper than three full models.
3. Full-model DMD2 only after LoRA path is validated, because DMD2 multiplies memory and checkpoint pressure.

Recommendation: use option 2 first.

## Training Data Path

DMD2 needs two streams:

1. Prompt/condition stream for generator sampling.
2. Real latent stream for GAN/classifier loss.

For FireRed, both can come from the same JSONL, but they must produce different tensors:

- generator stream: source image latent, conditional embedding, uncond embedding, random noise;
- real stream: target gray latent, same condition, source image latent.

Precompute target/source latents before long runs. Do not encode PNGs inside every training iteration.

## First FireRed DMD2 Dryrun

Use `configs/firered_gray_dmd2_lora.yaml`.

Current local data note: `cos_data_clay` currently contains a small local mirror, not the full dataset. This is enough for one-batch dryrun and short fastrun if we materialize a small local JSONL, but it is not enough for a full training run over `clay.jsonl`.

Required checks:

- JSONL exists and has valid fields.
- source/target images resolve locally.
- offline embedding files resolve locally.
- original gray LoRA adapter exists.
- model directories exist.
- `cfg_scale` is exactly `0`.
- output/checkpoint directory has enough free space.

Current status:

- 20-record local subset preflight passes.
- 3-step single-GPU local DMD2 LoRA dryrun passes.
- Passing run: `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/outputs/dmd2_firered_gray_lora_smoke/local_3step_20260622_072159`

Then run in this order:

1. Teacher-only inference on 2 samples.
2. Student/fake-critic model construction. Done for local dryrun.
3. One forward pass with `no_save`. Done through real training step.
4. One generator update and one guidance update. Done.
5. 20-step local fastrun.
6. 100-step local or Slurm fastrun with full comparison contact sheet.

## Evaluation Format

Keep the same contact sheet format used for TwinFlow:

```text
input | orig_lora_vanilla_40 | orig_lora_few_1nfe | dmd2_1nfe | dmd2_few_2nfe | dmd2_few_4nfe | target
```

For the first DMD2 comparison, `dmd2_few_4nfe` is the primary target, because upstream DMD2 4-step is the most stable path.

## Known Risks

- QwenImageEdit is not an epsilon-prediction UNet; DMD2's `get_x0_from_noise` must be replaced, not reused blindly.
- FireRed edit conditioning includes source-image latents; score matching and classifier loss must be conditioned on the same source/prompt pair.
- Full-model DMD2 would likely exceed our current checkpoint/storage margin.
- Upstream DMD2 saving is weak for resume under FSDP; our FireRed implementation must keep the stronger resume/offline-eval logic we added for TwinFlow.
- The current local DMD2 dryrun saves LoRA adapters and a latent realism head, but it does not yet implement full optimizer/RNG resume.
