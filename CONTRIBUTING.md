# Adding a tool

A tool is built because a question needs it. Before you start, `ls tools`: a sibling may be one size away from what you need.

## The layout

Copy the nearest sibling's directory to `tools/<name>/`, where `<name>` is lowercase words joined by hyphens, and change what differs.

- `SKILL.md`: the skill, the tool's whole description.
- `Containerfile`: the image.
- `test/run.sh`, with a tiny fixture beside it: exits 0 iff the tool works.
- `bin/`: the wrapper program, only where upstream ships no command-line program.

## The conventions

- **The environment first, the weights last**, then `COPY test /toolshed/test` as the last line. Size variants then share their base layers.
- **Weights are fetched at build time, never at run time.** Set `HF_HUB_OFFLINE=1` in the image, and `TRANSFORMERS_OFFLINE=1` where transformers is used.
- **Pin everything**: the base image and every package, at the versions current when you build.
- **One image per size variant.** `esm2-650m` and a future `esm2-3b` are two tools; the second is a copy of the first with the weights line and the skill's numbers changed.
- **A wrapper only where upstream has none.** Where upstream is a library, `bin/` holds one short program exposing what the lab wants as subcommands (`esm2 pll`, `esm2 embed`). Where upstream ships a program (`boltz predict`), the image makes it runnable and the skill says which commands matter.
- **Every example in the skill is a full command line**: the program on the image's PATH with its arguments. Pyxis ignores entrypoints.

## The skill

Frontmatter has three fields a machine reads:

```markdown
---
name: <name>
description: <what it is for, and when to reach for it>
metadata:
  version: 1.0.0
---
```

`name` is the directory's name, `description` is the line the catalogue shows, and `metadata.version` is the image tag. The frontmatter is YAML: quote the description in double quotes when it holds a colon. Then the body, in this order: `Image:` (the full reference with its version), `## Commands`, `## Inputs and outputs`, `## Compute` (GPUs, memory, a measured rate), `## Running it` (the laptop line and the cluster line, with real commands), `## Gotchas`. No declared parameter lists: prose, until agents are seen misusing commands from it.

## The test

`test/run.sh` runs the commands the skill documents on a fixture of kilobytes with a known answer, and checks the answer. It exits 0 iff the tool works, writes only under `/tmp`, passes with no network, and fits in one GPU and 30 minutes. It checks the environment, the weights, and that the skill's examples are true, in one step.

## Building and testing locally

```bash
scripts/build <name> ghcr.io/brineylab/toolshed/<name>:dev
scripts/test ghcr.io/brineylab/toolshed/<name>:dev
python3 -I scripts/check.py tools/<name>
```

`scripts/test` runs the test with `--network none`, with the GPU when the machine has one.

## Versions

Any change under `tools/<name>/` bumps `metadata.version`, prose included. A published tag is never overwritten, and CI refuses a pull request whose version is already published.

## The pull request

1. Push a branch to this repository, not a fork: a fork's pull request is checked but not built.
2. CI checks the directory and builds the candidate, `ghcr.io/brineylab/toolshed/<name>:pr-<n>`; the build's summary names it.
3. Test the candidate where you will use it. CI has no GPU and no cluster, so this is yours:

   ```bash
   srun --partition=<partition> --gres=gpu:1 --time=00:30:00 \
     --container-image=ghcr.io#brineylab/toolshed/<name>:pr-<n> bash /toolshed/test/run.sh
   ```

   or `scripts/test ghcr.io/brineylab/toolshed/<name>:pr-<n>` on a laptop.
4. Paste the output into the pull request, with where it ran.
5. Someone other than you reviews it: does the skill describe what the Containerfile builds, does the test exercise every documented command, are the weights in the image, is every example a full command line, did the test pass where it ran. They merge with a merge commit, and CI publishes the version.

## Weights behind a login

For a gated Hugging Face repository, the build takes the token as a secret, never as an argument or a layer:

```dockerfile
RUN --mount=type=secret,id=hf_token HF_TOKEN=$(cat /run/secrets/hf_token) HF_HUB_OFFLINE=0 python -c "..."
```

Locally, export `HF_TOKEN` before `scripts/build`. In CI, the repository secret `HF_TOKEN` is passed to the runner.
