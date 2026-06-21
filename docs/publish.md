# Publish Notes

Chosen repository name:

`CHDTevior/dmd2-firered-porting-harness`

Chosen visibility:

`private`

Reason: this harness contains internal absolute paths, Slurm assumptions, and local dataset mirror details. It should not be public until paths and internal environment references are sanitized.

## Current State

This directory is already a local git repository with a clean initial commit.

If the GitHub repository already exists, push with:

```bash
git remote add origin git@github.com:CHDTevior/dmd2-firered-porting-harness.git
git push -u origin master
```

If the remote has already been configured:

```bash
git push -u origin master
```

## If GitHub Repo Does Not Exist

Create a private empty repo named `dmd2-firered-porting-harness` under `CHDTevior`, then push from this directory.

The current machine has SSH authentication to GitHub as `CHDTevior`, but it does not have `gh` installed and no `GITHUB_TOKEN`/`GH_TOKEN` is available. Therefore repo creation could not be performed from this shell without an additional GitHub creation interface.

## Bundle Export

A portable bundle can be created with:

```bash
git bundle create /vepfs-cnbja62d5d769987/suntengjiao/distill/dmd2-firered-porting-harness.bundle --all
```

Import elsewhere with:

```bash
git clone dmd2-firered-porting-harness.bundle dmd2-firered-porting-harness
```

