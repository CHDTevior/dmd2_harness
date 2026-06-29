# Publish Notes

Chosen repository name:

`CHDTevior/dmd2_harness`

Chosen visibility:

`private`

Reason: this harness contains internal absolute paths, Slurm assumptions, and local dataset mirror details. It should not be public until paths and internal environment references are sanitized.

## Current State

This directory is already a local git repository with a clean initial commit.

The GitHub repository exists at:

`git@github.com:CHDTevior/dmd2_harness.git`

Push with:

```bash
git remote set-url origin git@github.com:CHDTevior/dmd2_harness.git
git push -u origin master
```

## Bundle Export

A portable bundle can be created with:

```bash
git bundle create /vepfs-cnbja62d5d769987/suntengjiao/distill/dmd2_harness.bundle --all
```

Import elsewhere with:

```bash
git clone dmd2_harness.bundle dmd2_harness
```

## GitHub Target

GitHub target name:

`CHDTevior/dmd2_harness`

The repository is intended to stay private/internal because configs and docs include cluster paths, Slurm queue names, and FireRed/TwinFlow integration assumptions.

Before pushing:

```bash
find . -type f \( -size +25M -o -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' \) -print
python -m py_compile scripts/*.py src/dmd2_firered/*.py
git status --short
```

Expected large-file scan output is empty.
