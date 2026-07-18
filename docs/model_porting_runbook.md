# Model Porting Runbook

This is the reusable checklist for taking a new conditional image generation or
image editing model from "not supported" to a reproducible DMD2 experiment.
Keep this repository small: commit code, config templates, tiny smoke inputs,
manifests, and documentation. Do not commit weights, full datasets,
checkpoints, W&B runs, or generated evaluation directories.

## 1. Define The Contract Before Code

Write the following decisions in the config before implementing the adapter.

| Decision | Required answer |
| --- | --- |
| Student form | LoRA or full model; which parameters are trainable? |
| Teacher capability | frozen base, frozen adapter, or merged teacher? |
| Real/fake topology | separate fake critic or an explicitly justified alternative? |
| Student training NFE | rollout NFE used by the DMD2 loss |
| Student eval NFE | exact sampler plus tested NFE values |
| CFG contract | real teacher scale, fake critic scale, student scale |
| Resolution | train and eval resolution |
| GAN feature | model layer, feature shape, and all conditioning inputs |
| Checkpoint mode | inference-only or exact continuation |
| Slurm topology | GPUs per node, node count, and expected free storage |

For the FireRed gray target, the canonical choices are full student, separate
full fake critic, merged gray teacher, `dmd2_renoise`, `real CFG=4`, `fake
CFG=1`, `student CFG=0`, bf16 FSDP, and 1024 evaluation every 250 steps.

## 2. Make The Config The Source Of Truth

Keep the following groups explicit:

- `model`: model family, merged teacher path, text-encoder behavior, dtype.
- `data`: JSONL, image fields, cond/uncond embeddings, resolution.
- `method`: DMD2 topology, losses, NFE, CFG, GAN feature layer, update ratio.
- `sample` and `eval`: sampler, student CFG, NFE, reference manifest.
- `train`: optimizer, FSDP, checkpoint mode, retention, and output path.

Use fail-fast validation. A distributed job must reject a legacy sampler,
missing unconditional embedding, unsupported checkpoint policy, or wrong NFE
before allocating hours of GPUs.

## 3. Build A Condition-Complete Dataset Adapter

For an image-editing model each batch needs all of the following, aligned to
the same record identity:

- source image or source latent;
- target image or target latent;
- conditional prompt representation;
- unconditional prompt representation when the real teacher uses CFG;
- an identifier used by the fixed evaluation manifest.

Never replace missing images, embeddings, or remote paths with a blank value.
Raise a row-specific error in preflight. For FireRed this is the source image,
target image, `embeddings_tensor_en`, and `embeddings_tensor_droptext`.

## 4. Implement The Model Wrapper From Native Semantics

The adapter must define an unambiguous call such as:

```python
velocity = model_fn(x_t, t, [prompt_embeds, prompt_mask, source_latents])
```

Do not copy an epsilon-diffusion formula into a flow model. For the FireRed
velocity parameterization the predicted clean latent is:

```text
x0 = x_t - t * velocity
```

Pass the same source and prompt condition to student, real teacher, fake
critic, GAN classifier, and evaluation. Otherwise the score comparison is not
defined on the same conditional distribution.

## 5. Use The Official DMD2 Topology

The current FireRed path is `DMD2FullOfficial`:

- trainable student/generator;
- frozen real teacher queried at `real_guidance_scale > 1`;
- trainable separate fake critic queried at `fake_guidance_scale: 1`;
- optional Qwen hidden-state GAN classifier on the fake critic.

The distribution-matching loss is not a generic flow-matching loss and it is
not equivalent to reducing the inference step count. The student rollout uses
`method.student_train_sampling_steps`; the real and fake score terms are then
evaluated on noisy student samples.

For the Qwen GAN port, attach the classifier to a middle fake-critic block.
The head must consume edit tokens, source tokens, pooled prompt condition, and
timestep. A latent-only head discards the model's conditioning path and is not
equivalent to upstream's bottleneck classifier.

## 6. Separate Source Baseline From Distilled Evaluation

The original source pipeline and the DMD2 student have different samplers.

| Column | Correct protocol |
| --- | --- |
| Source baseline | FireRed source `FlowMatchEulerDiscreteScheduler`, 40 NFE, CFG 4.0 |
| DMD2 student | `dmd2_renoise`, configured NFE, CFG 0 |
| DMD2 teacher | guided score query during training, not a student inference column |

Do not label a source-scheduler 4-step output as a DMD2 4-NFE result. Keep
the source baseline as a separate manifest-driven reference column.

Use the same fixed four records, source images, prompts, seed, and 1024
resolution at every checkpoint. Keep machine-readable manifests. In a human
report, render independent images side by side for one conclusion per row; do
not use a relabeled concatenated contact sheet as evidence.

## 7. Preflight Before Slurm

The preflight must validate:

- model directory and transformer files;
- train/eval JSONL and all required fields;
- representative source image and embeddings;
- local/COS access mode;
- sampler and CFG contract;
- eval/save cadence and checkpoint retention;
- expected disk space at the output path.

Run it again inside the Slurm allocation because node-local paths,
environment, and COS credentials can differ from the submit host.

## 8. FSDP And Slurm

For FireRed 1024 full DMD2 use two nodes of eight GPUs and bf16 FSDP.

```bash
sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:8 \
  --partition=<site-gpu-partition> \
  scripts/sbatch_firered_dmd2_full_fsdp.sh <config.yaml>
```

Passing only `--nodes=2` is incorrect when the script declares one task: Slurm
can collapse the allocation to one node. The launcher must resolve IPv4 for
the master and each local rank. Hostname IPv6 warnings are secondary; the
success condition is two `[LaunchNode]` entries with routable IPv4 addresses.

Use `gradient_checkpointing_use_reentrant: false`. Reentrant activation
checkpointing can replay a forward segment in an order that conflicts with
FSDP's expected module execution order. The symptom is a forward-order or
collective hang, not a useful training slowdown.

## 9. Choose Checkpoints Deliberately

`model_only_eval` saves only student model shards and metadata. It is suitable
for periodic visual evaluation with one retained checkpoint, but cannot resume
DMD2 because the fake critic, GAN classifier, optimizers, cursor, and RNG are
missing.

`full_training_state` saves all of those components and enforces same-world
size loading and an identical resolved-config fingerprint. It needs storage for the completed old state and the new state
being written. Do not delete the only completed state before a replacement has
the completion marker. This is a correctness rule, not just a retention
preference.

Use a dedicated, deterministic DataLoader generator. Creating a new iterator
after restoring process RNG otherwise consumes Torch's global RNG to seed data
workers and breaks bitwise continuation before the next training step.

## 10. Required Gates

1. Upstream DMD2 smoke passes.
2. Data/model/config preflight passes.
3. One-step local runtime smoke passes.
4. Two-node 1024 smoke shows finite student/fake/GAN losses.
5. First scheduled 250-step eval produces the fixed reference comparison.
6. Only then start a long run or sweep.

For a `dfake_gen_update_ratio` of 5, four fake-critic updates followed by one
student update are expected. Zero student gradient on critic-only iterations
is normal; a nonfinite value or an unexpected generator cadence is not.
