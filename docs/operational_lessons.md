# FireRed DMD2 Operational Lessons

This document records issues found while moving DMD2 from upstream SDXL to the
FireRed/QwenImageEdit gray-edit model. Each item is a guardrail for later
ports, not a claim that every diffusion model has the same implementation.

## 1. Legacy Shared-Fake Configs Were Not Official DMD2

Early FireRed configs used `DMD2FullShared`, a `few` sampler, flow-matching
losses, and a CFG-bake term. That path was useful for smoke debugging but it
was not the supported full-official DMD2 implementation. The runnable full
path now requires:

```yaml
method:
  method_type: DMD2FullOfficial
  critic_mode: separate_full
  real_guidance_scale: 4.0
  fake_guidance_scale: 1.0
  student_train_sampling_steps: <nfe>
sample:
  cfg_scale: 0
  sampling_style: dmd2_renoise
```

The trainer validates these fields and rejects the legacy sampler/topology.

## 2. CFG Is Distilled Through The Real Teacher

The source FireRed reference uses 40 NFE with CFG 4.0. In DMD2 training, the
real teacher is queried with the configured positive CFG scale, while the fake
critic stays at scale 1. The student is evaluated with CFG 0, without an extra
positive/negative two-branch pass. The unconditional embedding is still
required because the guided real-teacher score needs it.

Treat CFG=0 as an explicit one-branch student evaluation policy, not as an
instruction to use the unconditioned prompt.

## 3. The DMD2 Student Does Not Use The Source 4-Step Scheduler

The source pipeline's `FlowMatchEulerDiscreteScheduler` is only the
non-distilled baseline. DMD2 trains and evaluates the student with
`dmd2_renoise`, which re-noises the student rollout to reduce train/inference
mismatch. Comparing a source-scheduler 4-step image against a DMD2-renoise
4-NFE image is valid as a quality comparison, but they are different sampling
protocols and must be labeled as such.

## 4. The GAN Head Must Preserve Qwen Conditions

Upstream DMD2 attaches a classifier to a UNet bottleneck. FireRed/Qwen has no
UNet bottleneck with the same public interface. The port hooks a configured
middle Qwen transformer block of the fake critic and splits its hidden state
into edit and source token grids. The classifier also receives pooled prompt
embeddings and diffusion time.

The old latent-only classifier was not equivalent: it could not see the full
source/prompt-conditioned representation. The new implementation fails fast
when the block feature, token layout, prompt shape, or feature layer is wrong.

## 5. FSDP Requires Non-Reentrant Activation Checkpointing

Two-node 1024 training requires bf16 FSDP. With reentrant activation
checkpointing, recomputation can revisit wrapped modules in an order that does
not match FSDP's forward-order bookkeeping. This led to order-check failures
or apparent distributed stalls. Use:

```yaml
train:
  precision_mode: bf16
  gradient_checkpointing: true
  gradient_checkpointing_use_reentrant: false
```

## 6. Two Nodes Need Two Slurm Tasks

The batch script defaults to one task for local/single-node smoke. A two-node
call must override nodes, task count, tasks per node, and GPUs together. The
correct shape is `--nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:8`.

The launcher resolves IPv4 addresses to avoid a hostname selecting an
unsupported IPv6 address family. The IPv6 warning alone is not the root cause
if `MASTER_ADDR`, `--local_addr`, and the two launch-node addresses are IPv4.

## 7. Resume And Disk Retention Conflict

Full DMD2 continuation needs student, fake critic, GAN, optimizer, cursor,
and per-rank RNG. A student-only checkpoint cannot reconstruct that state.

If exact continuation matters, retain the previous completed full state until
the new state has completed; this temporarily needs storage for both. If disk
space rules that out, choose `model_only_eval`, explicitly disable resume, and
retain only a student inference checkpoint. Never label the latter as
resumable.

Exact continuation also requires an unchanged resolved config fingerprint and
a dedicated deterministic DataLoader generator. Recreating a DataLoader
iterator with the global Torch generator after RNG restoration advances the
training RNG before the next batch, so the full trainer supplies a separate
epoch/rank generator and rejects changed resume configs.

## 8. Scheduled Evaluation Is A Training Gate

At 1024, retain each 250-step visual evaluation even when only one checkpoint
is retained. Use a fixed four-record, 1024, source-CFG4 baseline reference and
student CFG0 DMD2-renoise columns. For conclusions, put independent PNG files
in horizontal rows so that one row answers one visual question such as detail
retention or background cleanliness.

## 9. Read The DMD2 Update Schedule Correctly

With `dfake_gen_update_ratio: 5`, the fake critic updates every iteration and
the student/generator updates once every five. Logs therefore show zero
student loss and zero student gradient on critic-only iterations by design.
Track `gen_update`, finite values, GAN real/fake logits, and step time rather
than treating those zeros as a failure.
