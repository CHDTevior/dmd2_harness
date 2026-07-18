# D-DMD FireRed Implementation

## Source Contract

The implementation follows the decomposition in
[Decoupled DMD](https://arxiv.org/html/2511.22677v1):

```text
Delta_DM = fake_cond - teacher_cond
Delta_CA = -(alpha - 1) * (teacher_cond - teacher_uncond)
Delta = Delta_DM + Delta_CA
```

The signs above are in this harness's x0 proxy-gradient convention,
`fake_x0 - teacher_cfg_x0`. When DM and CA use the same re-noised latent and
time, the sum is exactly the previous coupled DMD2 gradient.

The public [Z-Image repository](https://github.com/Tongyi-MAI/Z-Image) provides
inference code and model context but no D-DMD training implementation. The
paper algorithm and the existing verified FireRed DMD2 trainer are therefore
the implementation authorities.

## Time Convention

The paper uses `tau=0` for noise and `tau=1` for clean data. FireRed uses:

```text
x_t = t * noise + (1 - t) * x0
```

so FireRed has `t=1` for noise and `t=0` for clean data. The paper's focused
CA condition `tau_CA > t_generator` consequently becomes:

```text
t_CA < t_generator
```

For the first implementation:

- `t_DM ~ U(0.02, 0.98)` independently of the selected student stage;
- `t_CA ~ U(0.02, min(0.98, t_generator))`;
- DM and CA use independent Gaussian re-noising in production;
- `alpha` is the explicit `ca_guidance_scale`; the baseline pins it to
  `real_guidance_scale`, currently 4.0.

`decoupled_ca_mode: full` samples CA independently over its configured full
range (paper config ②). `constrained` applies the flipped-time inequality above
(paper config ④ and the production baseline).

For a four-step single-step student, the selected generator times are
`1.0, 0.75, 0.5, 0.25`. Configuration validation rejects a CA lower bound
that would make the final interval empty.

## What Is Unchanged

- full student, frozen merged-gray teacher, and separate full fake critic;
- fake critic denoising update and `dfake_gen_update_ratio: 5`;
- detached MSE proxy, combined-gradient normalization, clipping, and weights;
- Qwen middle-hidden-state GAN head with source, prompt, and timestep context;
- bf16 FSDP and non-reentrant activation checkpointing;
- DMD2 re-noise student sampling, inference CFG 0, and 1/2/4-NFE eval;
- one model-only checkpoint retained, with visual eval every 250 steps.

Keeping GAN enabled in both control and treatment tests D-DMD on the
GAN-regularized FireRed port. It is not a clean claim that GAN is required by
the D-DMD paper.

## Verification Gates

The checked-in tests cover:

1. forced-tie raw-gradient equivalence;
2. forced-tie normalized-gradient equivalence;
3. explicit one-batch time/noise integration without an optimizer step;
4. constrained CA bounds at all four student stages;
5. trainer-level backward with three teacher calls, two fake calls, finite
   student/fake gradients, and no CA bound violation;
6. trainer-level forced-tie coupled/decoupled equivalence for `loss_dm` and the
   student gradient.

Production logs add `t_gen`, `t_dm`, `t_ca`, `delta_dm`, `delta_ca`, and
`ca_violation`. The violation guard uses a distributed MAX reduction and aborts
all ranks if any rank violates its mode's interval. Checkpoint metadata and the
final run manifest record the decoupling mode, effective alpha, and all
timestep bounds.

## First Baseline

| Setting | Value |
| --- | --- |
| Student | full merged-gray FireRed transformer |
| Student training | 4 NFE, `single_step` |
| Teacher / CA CFG | 4.0 |
| Student inference CFG | 0 |
| Regularizers | DM + CA + Qwen hidden-state GAN |
| Train / eval resolution | 1024 / 1024 |
| Optimizer LR | student `5e-6`, fake `5e-6` |
| Batch | 1 per GPU, 16 A800 GPUs, global batch 16 |
| Precision | bf16 |
| Length | 3000 iterations |
| Eval / save | every 250; retain one model-only checkpoint |
| Primary eval | DMD2 re-noise 4 NFE; also 1 and 2 NFE |

Use the source 40-NFE CFG4 image only as the labelled non-distilled reference.
The efficacy comparison is the coupled DMD2 control versus this D-DMD
treatment under otherwise matched settings.
