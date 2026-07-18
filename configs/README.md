# Config Selection

Use only the `firered_gray_dmd2_full_official_*.yaml` configs for the current
FireRed/Qwen full-model DMD2 path. They require `DMD2FullOfficial`,
`critic_mode: separate_full`, `dmd2_renoise`, student/eval CFG 0, and the
250-step save/eval cadence enforced by the trainer.

The files named `*_full_shared*`, `*_full_cfg*`, and `*_lora*` predate the
full-official implementation. They are retained as historical smoke/debug
records and are intentionally rejected by the current full trainer. Do not use
them as templates for a new experiment.

Choose one checkpoint policy explicitly:

- `model_only_eval`: single student inference checkpoint; no resume.
- `full_training_state`: exact resume only when there is enough space for both
  the existing completed state and the new state being written. It must set
  `checkpoint_preclean_before_save: false`; pruning happens only after the new
  state receives its completion marker.
