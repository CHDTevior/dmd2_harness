# Model Porting Runbook

This document is the reusable checklist for taking a new image generation or image editing model from "not supported" to "DMD2 trainable" in this harness.

The repository should stay lightweight. Commit code, configs, tiny smoke inputs, manifests, and contact-sheet examples. Do not commit model weights, optimizer checkpoints, full datasets, W&B runs, or generated evaluation directories.

## 1. Define The Distillation Target

Before writing training code, make the target explicit:

| Decision | Required answer |
| --- | --- |
| Student form | LoRA adapter or full-model checkpoint |
| Teacher | frozen base, frozen LoRA, or merged full teacher |
| Inference NFE | usually `few 1`, `few 2`, or `few 4` |
| Training NFE | the student rollout used inside DMD2, not just eval |
| CFG policy | whether CFG is used in teacher/training and whether eval uses CFG |
| Resolution | training resolution and evaluation resolution |
| Resume policy | whether optimizer, train state, RNG, and world size must match |
| Checkpoint retention | how many checkpoints are kept on shared storage |

For the FireRed gray full-model experiment:

- student form: full FSDP transformer checkpoint;
- teacher capability: original gray LoRA merged into the FireRed backbone before DMD2;
- training NFE: `4`;
- training CFG: `4.0` through detached teacher target / CFG bake loss;
- evaluation CFG: `0`;
- evaluation variants: `few 1`, `few 2`, `few 4`;
- save/eval cadence: every `500` iterations;
- checkpoint retention: latest `1` for full-model runs.

## 2. Add A Config First

The config is the source of truth. Avoid hidden constants in the training script.

Minimum config groups:

- `model`: model family, model path, text encoder behavior, dtype assumptions.
- `data`: JSONL files, image fields, conditional/unconditional embedding fields, resolution.
- `method`: DMD2 losses, student rollout NFE, CFG training policy, fake critic policy.
- `sample`: default inference settings.
- `eval`: contact-sheet variants, reference manifest, cadence, sample count.
- `train`: optimizer, precision, FSDP settings, save/resume settings, output path.

Fail fast if a required field is missing. A long Slurm job should not discover a bad JSONL key after GPUs are allocated.

## 3. Build The Dataset Adapter

DMD2 needs condition-aligned real and generated samples. For image editing models this means each batch must contain:

- source image or source latent;
- target image or target latent;
- conditional prompt representation;
- unconditional prompt representation if CFG is trained or evaluated;
- record identity for evaluation manifests.

For FireRed, the adapter uses `FireRedEditJsonlDataset` from TwinFlow and reads:

- `source_image`;
- `edit_image`;
- `embeddings_tensor_en`;
- `embeddings_tensor_droptext`.

Do not silently replace missing images or embeddings. Raise an explicit error with the row number and missing field.

## 4. Wrap The Model

The DMD2 method only needs a small model API:

```python
velocity = model_fn(x_t, t, [prompt_embeds, prompt_mask, source_latents])
```

Porting a new model means writing the wrapper that maps this call into the model's native forward pass.

For non-epsilon diffusion models, do not reuse SDXL formulas blindly. FireRed/QwenImageEdit is a flow/velocity model, so the predicted clean latent is:

```text
x0 = x_t - t * velocity
```

If the model has image-edit conditioning, the source condition must be present in every teacher, student, fake-critic, and eval call. A score comparison with mismatched conditions is invalid.

## 5. Choose The DMD2 Topology

There are two supported experiment shapes:

| Topology | Use case | Storage/memory profile |
| --- | --- | --- |
| LoRA student + LoRA fake critic | first port, cheap ablations, capability-specific distillation | light checkpoints, easier iteration |
| Full student with shared/full fake path | final capability comparison, no adapter constraint | expensive FSDP checkpoints, needs strict retention |

The full FireRed experiment uses `DMD2FullShared`. It is LoRA-free and saves a full FSDP transformer checkpoint. Three independent full Qwen transformers are not practical on the shared storage/GPU setup, so the implementation keeps the fake-score path shared and controlled by the DMD2 loss design.

## 6. Implement Training Losses

The FireRed full-model implementation keeps four explicit loss terms:

- `fm`: flow matching on real target latents;
- `dm`: DMD2 distribution matching surrogate;
- `fake`: optional fake-score denoising loss;
- `cfg_bake`: optional loss that bakes a CFG teacher target into the conditional student.

The DMD2 update is not just "run fewer inference steps." The training step must sample the student according to `method.student_train_sampling_steps`, then compute the score/target correction used by the DMD2 surrogate.

Current full-model config uses:

```yaml
method:
  student_train_sampling_steps: 4
  student_train_backprop_mode: single_step
  train_cfg_scale: 4.0
  train_cfg_mode: teacher_detached
  cfg_bake_loss_weight: 0.1
sample:
  cfg_scale: 0
eval:
  cfg_scale: 0
```

## 7. Add Preflight Before Slurm

A good preflight checks:

- model directory and key model files exist;
- train/eval JSONL exists and has required keys;
- at least one source image and embedding can be loaded;
- config has the agreed `save_every` and `eval.every_steps`;
- checkpoint retention is bounded;
- free disk space is sufficient;
- COS/local data access is explicit.

The full FSDP sbatch script performs a second preflight after Slurm allocation because it also needs node-local environment and COS credentials.

## 8. Add Evaluation Before Long Training

Every port must produce the same style of contact sheet before long training:

```text
input | orig_lora_source_40_cfg4 | dmd2_full_1nfe | dmd2_full_few_2nfe | dmd2_full_few_4nfe | target
```

For FireRed gray, evaluation uses `CFG=0` for the distilled model. The original non-distilled reference can be generated separately with its source script and `CFG=4.0`, then passed in through `eval.reference_manifest`.

Keep contact-sheet images and JSON manifests. Do not keep all intermediate generated images unless they are needed for debugging.

## 9. Slurm / FSDP Notes

For 1024 full-model FireRed DMD2, the clean smoke path is:

```bash
sbatch -N 2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --gres=gpu:8 \
  -p gpu-a800-traing-queue-02-single \
  scripts/sbatch_firered_dmd2_full_fsdp.sh \
  configs/firered_gray_dmd2_full_cfg4_4nfe_1024_1step_smoke.yaml
```

Important FSDP choices:

- use bf16 mixed precision;
- use `gradient_checkpointing_use_reentrant: false`;
- bind `torchrun --local_addr` to the node IPv4 address;
- bind `dist.barrier(device_ids=[local_rank])`;
- keep `TORCH_DISTRIBUTED_DEBUG=DETAIL` out of production runs;
- keep `FIRERED_DISABLE_FLASH_ATTN=1` on this environment because the installed wheel requires a newer glibc.

The 2-node 1024 smoke passed with job `8382`, 16 A800 GPUs, and a single DMD2 training step. It completed without FSDP forward-order warnings, OOM, or fatal NCCL errors. One step took about `268s`, so 1024 training is feasible but expensive.

## 10. Debugging Rules

When a distributed run is slow or stuck, isolate one axis at a time:

1. config and data preflight;
2. one-node 512 smoke;
3. one-node 1024 smoke if relevant;
4. two-node 1024 one-step smoke;
5. only then submit a longer run.

Known FireRed-specific findings:

- c10d `Address family not supported by protocol` hostname warnings were noisy but not the root cause;
- `FSDP Forward order differs` was the dangerous warning;
- the fix was non-reentrant checkpointing plus cleaner production distributed flags;
- full `[B,1,S,S]` attention masks at 1024 are expensive, so the Qwen wrapper should skip mask construction when the mask is all ones.

## 11. Publish Checklist

Before pushing the harness:

```bash
git status --short
find . -type f \( -size +25M -o -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' \) -print
python -m py_compile scripts/*.py src/dmd2_firered/*.py
```

The expected large-file scan result is empty. If it is not empty, do not push until those files are removed or ignored.
