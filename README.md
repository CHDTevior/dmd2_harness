# DMD2 FireRed Porting Harness

This repo is a local migration harness for comparing DMD2 against our TwinFlow FireRed gray/clay experiments.

Status on 2026-06-22:

- Upstream DMD2 cloned at `/vepfs-cnbja62d5d769987/suntengjiao/distill/dmd2`.
- Official DMD2 SDXL 4-step LoRA smoke is passing.
- Smoke output is archived in `artifacts/official_sdxl_smoke/`.
- FireRed migration is not trained yet; this repo defines the first migration design, preflight, and run phases.

## Official Smoke

Output:

![DMD2 SDXL LoRA smoke](artifacts/official_sdxl_smoke/dmd2_sdxl_lora_4step_smoke.png)

Smoke manifest:

`artifacts/official_sdxl_smoke/manifest.json`

The smoke uses the official DMD2 SDXL LoRA path:

- base: local fp16 subset of `stabilityai/stable-diffusion-xl-base-1.0`
- LoRA: `tianweiy/DMD2/dmd2_sdxl_4step_lora_fp16.safetensors`
- scheduler: `LCMScheduler`
- steps: `4`
- guidance: `0`
- timesteps: `[999, 749, 499, 249]`

Re-run:

```bash
bash scripts/run_official_sdxl_smoke.sh
```

## What DMD2 Does

DMD2 trains two trainable components:

- `generator`: the few-step student model.
- `fake critic`: a trainable score model that estimates the distribution of current generator samples.

It also keeps a frozen teacher:

- `real teacher`: the original diffusion model score function, usually evaluated with a stronger CFG setting.

On each generator update, DMD2 samples an image/latent from the student and compares teacher score versus fake-critic score on a noisy version of that student sample. The score difference is converted into a distribution-matching gradient. In the SDXL setup, DMD2 also adds a GAN-style classifier head on real VAE latents versus generated latents.

For 4-step SDXL, DMD2 uses denoising training with backward simulation to reduce train/inference mismatch. For 1-step SDXL, upstream also uses an ODE regression pretraining stage.

Detailed notes:

- `docs/dmd2_method_notes.md`
- `docs/firered_migration_plan.md`
- `docs/code_audit.md`

## FireRed Migration Target

Our current FireRed gray setup:

- base model: `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/ckpts`
- original gray LoRA: `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/ckpts/adapter_gray/adapter`
- merged full model: `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/ckpts_gray_lora_merged_v1`
- dataset: `/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/clay_meta/clay/clay.jsonl`
- offline embeddings: `embeddings_tensor_en`, `embeddings_tensor_droptext`
- evaluation must keep `CFG=0`

Before a smoke/dryrun:

```bash
python scripts/preflight_firered_dmd2.py --config configs/firered_gray_dmd2_lora_smoke.yaml
```

Before full training:

```bash
python scripts/preflight_firered_dmd2.py --config configs/firered_gray_dmd2_lora.yaml
```

## Recommended Migration Phases

1. Keep upstream DMD2 smoke reproducible.
2. Add FireRed data/model preflight and one-sample teacher inference.
3. Implement FireRed DMD2 adapter interfaces: model wrapper, scheduler/x0 conversion, dataset, real-data latents, classifier head.
4. Run a single-batch no-save dryrun.
5. Run a 100-step fastrun with contact-sheet eval.
6. Only after visual eval works, submit the real Slurm run.

## Key Design Decision

Start with LoRA DMD2, not full-model DMD2.

Reason: full FireRed/Qwen-Image-Edit already has high checkpoint/storage pressure. DMD2 needs student, fake critic, and frozen real teacher. A naive full-model implementation would multiply memory pressure. LoRA DMD2 can represent the student and fake critic as separate trainable adapters on top of one or two frozen base copies, which is much more realistic on the current shared A800/VEPFS setup.
