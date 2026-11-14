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
- **Weights are fetched at build time, never at run time.** Set `HF_HUB_OFFLINE=1` in the image, and `TRANSFORMERS_OFFLINE=1` where transformers is used, except in a general-purpose image (DESIGN §1), which sets neither: its work reaches the network.
- **A tool never carries a credential**: not in a file, not as an `ARG` or `ENV`. `scripts/build` refuses one it sees and says where. Gated weights take the build secret (below), which never reaches the image.
- **A GPU tool sets `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`** in the image, on a glibc base. NVIDIA's CUDA images already do; a plain `python:*-slim` does not, and under pyxis it then sees no GPU though podman does.
- **A Python tool loads its model from a path in the image and runs under `python3 -I`**: download the weights with `snapshot_download(repo, revision=<commit>, local_dir=...)` and start the wrapper with `#!/usr/bin/env -S python3 -I`. Pyxis passes the caller's environment into the container, and their home with `~/.local`, so a caller's `HF_HUB_CACHE`, `PYTHONPATH`, or user packages would otherwise replace what was tested.
- **Pin everything**: the base image and every package, at the versions current when you build.
- **One image per size variant.** `esm2-650m` and a future `esm2-3b` are two tools; the second is a copy of the first with the weights line and the skill's numbers changed.
- **A wrapper only where upstream has none.** Where upstream is a library, `bin/` holds one short program exposing what the lab wants as subcommands (`esm2 pll`, `esm2 embed`). Where upstream ships a program (`boltz predict`), the image makes it runnable and the skill says which commands matter.
- **Every example in the skill is a full command line**: the program on the image's PATH with its arguments. Pyxis ignores entrypoints.

## Licenses

Images are public, so a tool publishes its weights. Apply these rules as written:

1. **A permissive license** on the weights (MIT, Apache-2.0, BSD, CC-BY): publish.
2. **No license on the weights**: publish when the official code repository's license is permissive, and say so in the skill.
3. **A license that allows redistribution under conditions**: publish with every condition met. Copy the upstream license, notice, and attribution files into the image; name any attribution the license requires in the skill; and if it requires a notice where the model runs, have the program print it to stderr each time it starts.
4. **A license that forbids redistribution**: do not build the tool. Say so to whoever asked for it (DESIGN's private-tool row covers what comes later).
5. **Every tool** states its weights' license in `SKILL.md`, under *Gotchas* or beside the weights' source, and copies the upstream license files into the image beside the weights.

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

**The fixture is small, but it must be the size the tool takes seriously.** A tool often has a threshold below which it takes a cheaper path: a fused kernel it only compiles past a certain length, a chunked loop it only enters past a certain batch, a database it only opens past a certain query. A fixture under every threshold tests a tool nobody will run. `boltz-2@1.0.0` passed its test on a tiny complex and was published, then failed on the first real antibody: past cuEquivariance's 100-residue crossover, Triton compiles a kernel at run time and needs `gcc`, which the image did not have. `1.0.1` carries a 198-residue fixture. So before you settle the fixture, read what the tool does differently as its input grows (sequence length, number of chains, batch size, file size) and put the fixture on the far side of the first such threshold, as small as that allows. Say in the skill's *Gotchas* which threshold the fixture crosses, so the next person knows what the test does and does not cover.

## Building, pushing, and testing

You build your tool's image, push it, and test it on the hardware it needs. No cluster is required. CI never builds or runs a tool image.

1. **Commit** the tool directory. The image's tag is the candidate, `<name>:src-<tree>`, where `<tree>` is git's hash of the committed `tools/<name>/`, so `scripts/build` refuses uncommitted changes there.
2. **Build:** `scripts/build <name>`. It builds for `linux/amd64` (the clusters' architecture, even on an Apple Silicon Mac), labels the image with this repository, and prints the candidate's name.
3. **Try it:** `podman run --rm --network none <candidate> bash /toolshed/test/run.sh`.
4. **Push:** once per machine, `podman login ghcr.io` as yourself with a GitHub token that has `write:packages` (`gh auth refresh -s write:packages`, then `gh auth token | podman login ghcr.io -u <you> --password-stdin`). `gh` stays signed in: a tool's first push asks this repository's `seed.yml` to create its package, public and linked, and waits for it. Then `scripts/push <name>`. It exits 0 when the candidate can be pulled with no login.
5. **Test it on the hardware it needs**: any linux/amd64 machine for a CPU tool, one with the GPU the tool uses for a GPU tool. Where it will be used later does not matter; the conventions above are what make an image behave the same everywhere. On your own machine:

   ```bash
   scripts/test <name> podman
   ```

   It runs the candidate by digest with no network, attaching the machine's GPU when `nvidia-smi` is there (the README's CDI line sets that up once). A cluster works too, from its login node and a clone of your branch: `scripts/test <name> srun --partition=<partition> --time=00:30:00`, adding `--gres=gpu:1` for a GPU tool. Either way it writes the record, `results/<name>/<version>.txt`: the digest, where it ran, the node and GPU it saw, the time, the output, and the exit code. If you lack the hardware (a GPU tool on a laptop without one), push the candidate and ask someone who has it to run this step from a clone of your branch; the record says where it ran.
6. **Commit the record:** `git add results/<name>/<version>.txt`.

`python3 -I scripts/check.py tools/<name> --registry ghcr.io/<owner>/<repo>` runs CI's check on your machine.

**An agent building a tool** follows the same steps. The push (step 4) and the pull request publish under its person's name, so it asks them first. It runs step 5 where it is when that machine has the hardware, and otherwise hands that one step to its person.

A change after testing changes the tree, so build, push, and test again. A prose-only change that rebuilds from cache to the same digest may copy its earlier record to the new version's path, since the same image was tested.

## Versions

Any change under `tools/<name>/` bumps `metadata.version`, prose included. A published tag is never overwritten, and CI refuses a pull request whose version is already published.

## The pull request

1. Push your branch and open the pull request.
2. `check` runs: the tool's rules, an unpublished version, a public amd64 candidate for the branch's source, and a record of that candidate's passing test.
3. A member other than you reviews it: does the skill describe what the Containerfile builds, does the test exercise every documented command, are the weights in the image, is every example a full command line, and does the record show the test ran on the hardware the tool needs (its node and GPU lines) and gave the known answers. They merge with a merge commit.
4. `promote` copies the candidate to `<name>:<version>`.

## Weights behind a login

For a gated Hugging Face repository, the build takes the token as a secret, never as an argument or a layer:

```dockerfile
RUN --mount=type=secret,id=hf_token HF_TOKEN=$(cat /run/secrets/hf_token) HF_HUB_OFFLINE=0 python -c "..."
```

Export `HF_TOKEN` before `scripts/build`, which passes it as the build secret. The token stays with whoever builds; it is never a GitHub secret. `scripts/build` refuses a Containerfile that declares `ARG HF_TOKEN` or `ENV HF_TOKEN`, since either would keep the token in the image; the mount is the only way.
