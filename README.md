# DMD2 FireRed Porting Harness

This repo is a lightweight migration harness for comparing DMD2 against our TwinFlow FireRed gray/clay experiments and for reusing the same porting pattern on future models.

Status:

- Upstream DMD2 cloned at `/vepfs-cnbja62d5d769987/suntengjiao/distill/dmd2`.
- Official DMD2 SDXL 4-step LoRA smoke is passing.
- Smoke output is archived in `artifacts/official_sdxl_smoke/`.
- FireRed local DMD2 LoRA dryrun is passing on a real A800 GPU.
- FireRed DMD2 LoRA and full-model implementations are present.
- Full-model FireRed DMD2 uses FSDP, bf16, 4-NFE student training, CFG-4 teacher target, and CFG-0 distilled eval.
- Two-node 1024 full-model smoke passed on 16 A800 GPUs with job `8382`.
- The current repository intentionally excludes weights, full datasets, checkpoints, and generated evaluation directories.

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
- `docs/model_porting_runbook.md`

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

Run the current local FireRed DMD2 dryrun on the A800 server:

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 suntengjiao@175.178.95.29 \
  'cd /vepfs-cnbja62d5d769987/suntengjiao/distill/dmd2_firered_porting_harness && \
   export CUDA_VISIBLE_DEVICES=0 && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   PY=/vepfs-cnbja62d5d769987/suntengjiao/anaconda3/envs/twin_flow_qwen/bin/python && \
   $PY scripts/train_firered_dmd2_local.py \
     --config configs/firered_gray_dmd2_lora_smoke.yaml \
     --steps 3 \
     --device cuda \
     --eval-samples 2 \
     --fake-updates-per-step 1'
```

Latest passing local run:

`/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/outputs/dmd2_firered_gray_lora_smoke/local_3step_20260622_072159`

Detailed run notes:

- `docs/local_dmd2_dryrun_20260622.md`

Run the current Slurm training job with the agreed QA sheet:

```bash
sbatch scripts/sbatch_firered_dmd2_lora_fastrun.sh
```

Current defaults:

- steps: `20000`
- checkpoint save: every `500` steps
- checkpoint retention: latest `5`
- checkpoint content: student LoRA, fake-critic LoRA, latent realism head, optimizer states, RNG states
- walltime: `5-00:00:00`

This Slurm run trains on the current 20-record local subset and writes QA eval in:

`<run_dir>/offline_eval_qa/global_step_020000/contact_sheet.png`

Before full training:

```bash
python scripts/preflight_firered_dmd2.py --config configs/firered_gray_dmd2_lora.yaml
```

Full-model FSDP smoke:

```bash
sbatch -N 2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --gres=gpu:8 \
  -p gpu-a800-traing-queue-02-single \
  scripts/sbatch_firered_dmd2_full_fsdp.sh \
  configs/firered_gray_dmd2_full_cfg4_4nfe_1024_1step_smoke.yaml
```

Full-model long-run template:

```bash
sbatch scripts/sbatch_firered_dmd2_full_fsdp.sh \
  configs/firered_gray_dmd2_full_cfg4_4nfe_1024_3k.yaml
```

## Recommended Migration Phases

1. Keep upstream DMD2 smoke reproducible.
2. Add FireRed data/model preflight and one-sample teacher inference. Done.
3. Implement FireRed DMD2 adapter interfaces: model wrapper, flow-to-x0 conversion, dataset, real-data latents, classifier head. Done for local dryrun.
4. Run a single-batch no-save dryrun. Done.
5. Run a 100-step fastrun with the full comparison contact sheet. Slurm script added.
6. Only after visual eval works, submit the real Slurm run.

## Repository Contents

Keep this repo small:

- keep: scripts, config templates, preflight checks, small JSONL smoke subsets, smoke contact sheets, manifests, migration docs;
- exclude: model weights, LoRA weights, full datasets, optimizer checkpoints, W&B runs, generated eval folders.

The reusable porting checklist is in `docs/model_porting_runbook.md`.

## Key Design Decision

Start with LoRA DMD2, not full-model DMD2.

Reason: full FireRed/Qwen-Image-Edit already has high checkpoint/storage pressure. DMD2 needs student, fake critic, and frozen real teacher. A naive full-model implementation would multiply memory pressure. LoRA DMD2 can represent the student and fake critic as separate trainable adapters on top of one or two frozen base copies, which is much more realistic on the current shared A800/VEPFS setup.
