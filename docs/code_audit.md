# Code Audit

## Post-Audit Status Update (2026-07)

The original audit identified two open items that have since been addressed in
the supported full-model path:

- The GAN classifier is now a Qwen middle-hidden-state head on the fake critic,
  conditioned on edit/source token grids, prompt, and timestep. It replaces
  the earlier latent-only placeholder.
- The supported full run is now `DMD2FullOfficial` with a separate full fake
  critic, frozen teacher, DMD2 re-noise sampler, explicit real/fake CFG
  scales, and two checkpoint modes with fail-fast resume checks.

Historical LoRA/shared-fake notes below remain useful for debugging context,
but they are not the current full-official training protocol. See
`docs/operational_lessons.md` and `docs/model_porting_runbook.md` for the
current contract.

## Upstream DMD2 Issues To Fix Before FireRed Training

1. `main/train_sd.py` assumes W&B online mode.
   - FireRed Slurm jobs should support offline/no-W&B mode.

2. `train_sd.py` creates timestamped output and cache dirs independently.
   - It calls `time.time()` twice, which can produce mismatched output/cache names.
   - FireRed should use one run ID generated once.

3. Upstream FSDP checkpointing saves model weights only.
   - It explicitly says optimizer state under FSDP is not handled.
   - FireRed training needs resumable train state: model, optimizer, scheduler, RNG, dataloader progress.

4. `scripts/download_hf_checkpoint.sh` downloads entire checkpoint folders including optimizer files.
   - For inference smoke, only model weights are needed.
   - For our disk pressure, use targeted downloads.

5. Data loading is SDXL-specific.
   - `SDImageDatasetLMDB` only stores `latents` and prompt strings.
   - FireRed needs source image condition, target latent, prompt embedding, uncond embedding, uid, and local path checks.

6. Classifier head assumes UNet bottleneck geometry.
   - `classify_forward` returns UNet hidden states.
   - FireRed needs a new representation hook from QwenImage transformer hidden states or latent patches.

7. `get_x0_from_noise` assumes epsilon prediction.
   - FireRed/Qwen scheduler conversion must be implemented from the model's actual prediction semantics.

8. Upstream 4-step warning from diffusers:
   - `LCMScheduler` warns that `[749, 249]` are not on the default scheduler timesteps.
   - Upstream README explicitly uses these timesteps, so this is acceptable for official smoke but should be recorded in manifests.

## Local Smoke Notes

`red_train` failed before inference because importing diffusers touched an incompatible `flash_attn` binary:

```text
GLIBC_2.32 not found
```

The official smoke was run successfully in `twin_flow_qwen`.

## FireRed Local DMD2 Notes

The first real A800 dryrun exposed two dtype boundaries that are now fixed in `scripts/train_firered_dmd2_local.py`:

1. The fake-critic generated-sample branch produced fp32 noisy latents while the QwenImageEdit weights were bf16.
2. The no-grad student generation branch needed bf16 autocast because Qwen attention uses an additive mask and SDPA requires the mask/bias dtype to match the query dtype.

The passing run confirms:

- local image and embedding loading works;
- gray LoRA can be loaded into `teacher_gray`, `student`, and `fake_critic`;
- student and fake-critic optimizers can step on one A800;
- `loss_dm` becomes non-zero after fake-critic updates.

## FireRed Engineering Rules

- No implicit fallback from local files to COS/HF during Slurm training.
- No `cfg_scale != 0` for gray-model eval.
- No full run before one-batch dryrun and 100-step fastrun.
- All downloads are separate commands with proxy variables unset.
- All eval outputs should write a manifest and contact sheet.
