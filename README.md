# DMD2 FireRed Porting Harness

This repository is the lightweight, runnable record of porting upstream DMD2 to
the FireRed/QwenImageEdit gray-edit capability. It is also the template for
porting another conditional flow or image-editing model without repeating the
same implementation mistakes.

The repository contains code, config templates, a tiny smoke subset, and
documentation. It intentionally excludes weights, datasets, checkpoints,
training logs, and full generated-evaluation directories.

## Current Supported Path

The supported FireRed path is `DMD2FullOfficial`:

- a trainable full student transformer;
- a separate trainable full fake critic;
- a frozen full real teacher initialized from the merged gray FireRed model;
- a Qwen transformer hidden-state GAN classifier on the fake critic;
- a guided real teacher (`real_guidance_scale > 1`) and an unguided fake critic
  (`fake_guidance_scale: 1`);
- DMD2 re-noise sampling for the distilled student at train-time and eval;
- bf16 FSDP on two 8-GPU nodes for 1024 training.

The older `DMD2FullShared`, `few`, and local LoRA paths are historical debug
artifacts. They are not the current official FireRed DMD2 protocol.

## What Has Been Verified

- The upstream SDXL 4-step LoRA smoke passes. Its artifact is in
  `artifacts/official_sdxl_smoke/`.
- The FireRed full-official path passes a two-node, 16-A800, 1024 runtime
  smoke with FSDP.
- The Qwen hidden-state GAN head is installed at a configured middle block and
  receives edit, source, prompt, and timestep condition.
- Distributed training fails fast on invalid sampler, CFG, checkpoint, or
  resume settings rather than silently falling back to the legacy path.

## Use The Harness

Set paths in a copied config for the local cluster. The two checked-in configs
are concrete FireRed templates:

- `configs/firered_gray_dmd2_full_official_cfg4_4nfe_1024_3k_lr5e6_dmd2renoise_gan.yaml`
  is the 4-NFE official GAN protocol with multi-NFE eval variants.
- `configs/firered_gray_dmd2_full_official_cfg4_gan_student1_infer1_1024_3k_lr5e7_modelonly_v1.yaml`
  is the model-only 1/1-NFE example. It saves inference weights only and is
  deliberately not resumable.

Run preflight before requesting GPUs:

```bash
python scripts/preflight_firered_dmd2.py \
  --config configs/firered_gray_dmd2_full_official_cfg4_4nfe_1024_3k_lr5e6_dmd2renoise_gan.yaml
```

For a 1024 two-node Slurm run, override every resource dimension rather than
only passing `-N 2`:

```bash
sbatch \
  --job-name=dmd2_firered_4nfe \
  --partition=<site-gpu-partition> \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --gres=gpu:8 \
  --time=2-00:00:00 \
  --export=ALL,MASTER_PORT=19611 \
  scripts/sbatch_firered_dmd2_full_fsdp.sh \
  configs/firered_gray_dmd2_full_official_cfg4_4nfe_1024_3k_lr5e6_dmd2renoise_gan.yaml
```

The sbatch wrapper resolves a routable IPv4 master address and obtains local
IPv4 addresses for each `torchrun` rank. A hostname-only IPv6 warning is noisy
but is not a successful two-node setup; check the emitted `[LaunchNode]` lines
and confirm that both nodes are present in `squeue`.

## Sampling And CFG Contract

There are two distinct protocols. Do not mix them.

| Purpose | Sampler | NFE | CFG |
| --- | --- | --- | --- |
| Original FireRed reference | source `FlowMatchEulerDiscreteScheduler` | 40 | 4.0 |
| DMD2 real teacher during training | DMD2 score query | n/a | `real_guidance_scale`, usually 4.0 |
| DMD2 fake critic during training | DMD2 score query | n/a | exactly 1.0 |
| Distilled student train/eval | `dmd2_renoise` | configured NFE | 0 |

The source scheduler is a baseline generator, not the DMD2 student sampler.
The full-official trainer rejects `few` and rejects nonzero student/eval CFG.

## Checkpoint Modes

Choose the mode before launching, then leave the resume settings consistent
with it.

| Mode | Contains | Can resume | Storage rule |
| --- | --- | --- | --- |
| `model_only_eval` | student FSDP model shards plus metadata | no | one retained model is appropriate; preclean is allowed |
| `full_training_state` | student, fake critic, GAN head, optimizers, cursor, and per-rank RNG | yes, same world size | reserve space for the old state and the new state; do not preclean the sole valid state before a replacement finishes |

An incomplete checkpoint or a model-only checkpoint is intentionally rejected
as a DMD2 resume source. This avoids silently resuming without fake-critic,
GAN, optimizer, or RNG state.

## Documentation

- `docs/model_porting_runbook.md`: reusable migration checklist.
- `docs/operational_lessons.md`: errors found during the FireRed port and the
  corresponding guardrails.
- `docs/dmd2_method_notes.md`: upstream DMD2 mapping.
- `docs/code_audit.md`: original port audit and status.

## Publish Guard

Before pushing, keep the repository free of weights and generated training
outputs:

```bash
python -m py_compile scripts/*.py src/dmd2_firered/*.py
find . -type f \( -size +25M -o -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' \) -print
git status --short
```
