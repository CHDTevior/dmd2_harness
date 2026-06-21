# Implementation Slots

This package is intentionally a scaffold until FireRed DMD2 dryrun code is added.

Planned modules:

- `dataset.py`: JSONL dataset returning source latent, target latent, prompt embedding, uncond embedding, uid.
- `modeling.py`: student, teacher, and fake-critic wrappers around QwenImageEdit.
- `scheduler.py`: FireRed/Qwen conversion replacing SDXL `get_x0_from_noise`.
- `losses.py`: DMD2 distribution matching and conditional GAN losses.
- `eval.py`: contact sheet eval matching the TwinFlow comparison format.
- `train.py`: one-batch dryrun, fastrun, resumable Slurm training.

Do not copy SDXL-specific assumptions into these modules without an explicit adapter boundary.

