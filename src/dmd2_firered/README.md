# FireRed DMD2 Utilities

The production full-model trainer remains in
`scripts/train_firered_dmd2_full_fsdp.py`. This package holds lightweight,
testable boundaries shared by the harness:

- `local_firered_data.py`: the historical local LoRA smoke dataset adapter.
- `decoupled_dmd.py`: explicit re-noising, constrained/full CA scheduling,
  coupled and decoupled x0-space gradients, and the shared normalizer.

Keep scheduler and parameterization math in pure tensor functions where
possible. FireRed uses velocity prediction and `x_t=t*noise+(1-t)*x0`; do not
copy epsilon/DDPM formulas into this package.
