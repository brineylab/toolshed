# AGENTS.md

The lab's toolshed: published models and pipelines as self-contained OCI images, one prose skill per tool. A tool is an image with no entrypoint and a command-line program on its PATH; the caller supplies the command, and inputs and outputs are files in a mounted directory. The same image runs under podman on a laptop and under pyxis on a Slurm cluster.

Read before changing anything:

- `docs/VISION.md`: what the shed is for and what it is not.
- `docs/DESIGN.md`: what a tool is, the repository, the workflow, review.
- `docs/plans/toolshed-a.md`: the plan that builds the shed. Work it task by task; its Global Constraints are binding.
- `CONTRIBUTING.md`: how to add a tool, step by step. The rules below are the binding summary.

`docs/VISION.md` and `docs/DESIGN.md` are approved by the lab. Do not edit them unless asked to. If the work shows the design is wrong, stop and say so.

## Layout

```
tools/<name>/SKILL.md        the skill: what the tool is for, how to call it, what it needs
tools/<name>/Containerfile
tools/<name>/test/run.sh     plus a tiny fixture; exit 0 iff the tool works
tools/<name>/bin/            the wrapper program, only where upstream has none
scripts/check.py             a tool directory's rules, and whether its version is published
scripts/build <name> <image>
scripts/test <image>
tests/test_check.py
.github/workflows/tools.yml  check, build (lab runner), promote
```

## Commands

```bash
python3 -I -m unittest discover tests          # check.py's unit test: the only test the shed has
python3 -I scripts/check.py tools/<name>       # a tool's rules; add --registry ghcr.io/<owner>/<repo> to check the version
scripts/build <name> ghcr.io/brineylab/toolshed/<name>:dev
scripts/test ghcr.io/brineylab/toolshed/<name>:dev   # the image's own test, --network none
```

## Rules a tool must keep

- **Weights are in the image**, fetched at build time, never at run time. `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1` with transformers) is set in the image. The test passes with `--network none`.
- **The Containerfile puts the environment first and the weights last**, then ends with `COPY test /toolshed/test`. Every package and base image is pinned.
- **Every example in a skill is a full command line**, the program on PATH with its arguments. Pyxis ignores entrypoints.
- **Any change under `tools/<name>/` bumps `metadata.version`**, prose included. A published tag is never overwritten; callers always name a version.
- **The skill's frontmatter is `name` (the directory's), `description`, `metadata.version`.** Everything else is prose. No declared parameter lists.
- **A size variant is its own tool** (`esm2-650m`, `esmc-600m`): copy the sibling's directory and change the weights line and the skill's numbers.
- **The test runs the commands the skill documents** on a kilobyte fixture with a known answer, writes only under `/tmp`, and fits in one GPU and 30 minutes.
- **The wrapper in `bin/` is the only code the shed writes per tool**, and only where upstream ships no command-line program.

## Rules the shed keeps

- `scripts/` is Python 3.11 stdlib (run with `python3 -I`) or bash with `set -euo pipefail`. No third-party package in the repository itself.
- Nothing in `scripts/` or `.github/` names the lab. The registry is `ghcr.io/${GITHUB_REPOSITORY,,}/<name>`, so a fork is its own shed with no edits.
- CI never runs a tool's test: it has no GPU and no cluster. Whoever asks for the tool runs the candidate (`<name>:pr-<n>`) where it will be used and pastes the output into the pull request. An agent cannot claim a tool works from a local build alone.
- No cluster ever builds an image. Builds happen on the lab's self-hosted runner, or locally with `scripts/build`.
- Every tool is reviewed by someone other than its builder before it merges.

## What not to build

A tool exists because a question needed it. Do not add tools, commands, or options speculatively. Not in the shed: datasets, analysis helpers (plotting, loaders), and anything that sequences tools. The items in `docs/DESIGN.md` §7 (a `toolshed` command, parameter schemas, private tools, CPU tests in CI, and the rest) are deferred until their stated trigger; do not build them early.

## Commits and pull requests

A commit message is one sentence stating the behavior. A tool arrives as a pull request from a branch in this repository (a fork's pull request does not build). Merge with a merge commit (`gh pr merge --merge`): `promote` runs on the merge commit, and flyloft watches that check.
