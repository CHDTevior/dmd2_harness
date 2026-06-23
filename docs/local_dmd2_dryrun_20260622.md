# FireRed DMD2 Local Dryrun 2026-06-22

## Result

Local single-GPU FireRed/QwenImageEdit DMD2 LoRA dryrun is passing on:

- host: `gz-az7-gpu-a800-02-train-plt-server`
- GPU: one A800 80GB through `CUDA_VISIBLE_DEVICES=0`
- environment: `/vepfs-cnbja62d5d769987/suntengjiao/anaconda3/envs/twin_flow_qwen`
- config: `configs/firered_gray_dmd2_lora_smoke.yaml`
- local dataset subset: `artifacts/local_subset/clay_local_subset.jsonl`

Run output:

`/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/outputs/dmd2_firered_gray_lora_smoke/local_3step_20260622_072159`

Contact sheet:

`/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/outputs/dmd2_firered_gray_lora_smoke/local_3step_20260622_072159/eval/global_step_000003/contact_sheet.png`

Checkpoint:

`/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/outputs/dmd2_firered_gray_lora_smoke/local_3step_20260622_072159/checkpoints/global_step_000003`

## Command

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 suntengjiao@175.178.95.29 \
  'cd /vepfs-cnbja62d5d769987/suntengjiao/distill/dmd2_firered_porting_harness && \
   export CUDA_VISIBLE_DEVICES=0 && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   export PYTHONUNBUFFERED=1 && \
   PY=/vepfs-cnbja62d5d769987/suntengjiao/anaconda3/envs/twin_flow_qwen/bin/python && \
   $PY scripts/train_firered_dmd2_local.py \
     --config configs/firered_gray_dmd2_lora_smoke.yaml \
     --steps 3 \
     --run-id local_3step_20260622_072159 \
     --device cuda \
     --eval-samples 2 \
     --fake-updates-per-step 1'
```

## Observed Metrics

| step | loss_student | loss_teacher | loss_dm | loss_fake | guidance_cls | reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.003419 | 0.000028 | 0.000000 | 0.015027 | 1.386702 | 49.02 |
| 2 | 1.141351 | 0.000028 | 1.137948 | 0.016839 | 1.386965 | 49.74 |
| 3 | 1.071838 | 0.000017 | 1.068428 | 0.006443 | 1.387452 | 50.03 |

`loss_dm` is zero at step 1 because `student`, `teacher_gray`, and `fake_critic` are all initialized from the original gray LoRA. After the first fake-critic update, `loss_dm` becomes non-zero, which verifies that the DMD2 distribution-matching path is active.

## Implemented Path

- Loads the FireRed/QwenImageEdit backbone from local `ckpts`.
- Loads the original gray LoRA into three adapters:
  - frozen `teacher_gray`
  - trainable `student`
  - trainable `fake_critic`
- Uses local source/target PNGs and offline prompt embeddings only.
- Keeps FireRed eval/training CFG at `0`.
- Uses bf16 autocast, TF32, fused AdamW, gradient checkpointing, and explicit path/GPU/loss checks.
- Saves separate student and fake-critic LoRA adapters plus a latent realism head.

## Known Limitations

- This is a local smoke/dryrun, not the final DMD2 training recipe.
- The local mirror currently only covers the 20-record subset, not full `clay.jsonl`.
- Fake-critic update ratio was set to `1` for the smoke to reduce turnaround time; the config default remains `5`.
- Full requested QA eval has been added as `scripts/eval_firered_dmd2_qa.py` and was smoke-tested on one sample.

## QA Eval Format

The QA contact sheet columns are:

```text
input | orig_lora_vanilla_40 | orig_lora_few_1nfe | dmd2_1nfe | dmd2_2nfe | dmd2_4nfe | target
```

Smoke-tested QA output:

`/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/outputs/dmd2_firered_gray_lora_smoke/local_3step_20260622_072159/offline_eval_qa_test/global_step_000003/contact_sheet.png`

The Slurm fastrun wrapper runs training first, then launches this QA eval as a separate Python process:

`scripts/sbatch_firered_dmd2_lora_fastrun.sh`

Current Slurm defaults are `20000` training steps, checkpoint every `500` steps, and retention of the latest `5` checkpoints. Checkpoints include optimizer and RNG state for resume.
