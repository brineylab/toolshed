# The toolshed: Design

Status: approved 2026-10-23. The plan is [plans/toolshed-a.md](plans/toolshed-a.md). The flyloft side (calling tools from a campaign, building one when the shed lacks it) is designed in flyloft's repository, `docs/design/toolshed.md`; §6 below is its summary.

## 1. A tool

A tool is **an OCI image whose entrypoint is nothing and whose PATH holds a command-line program**: the image is the tool's environment, dependencies, and weights, and the caller supplies the command. Inputs and outputs are files in a directory the caller mounts. Nothing about a tool is Python-specific or flyloft-specific. The same image runs under `podman run` on a laptop, `docker run` where that is what exists, and `--container-image` on a Slurm cluster with pyxis.

- **One image per tool, many commands.** Where upstream ships a command-line program (`boltz predict`), the image makes it runnable and the skill says which commands matter. Where upstream is a library (ESM-2 through `transformers`, ESM-C through `esm`), the tool directory carries one short script exposing what the lab wants as subcommands of one program (`esm2 pll`, `esm2 embed`, `esm2 fill`). That script is the only code the shed writes per tool, and only when needed.
- **A size variant is a tool.** `esm2-650m`, `esmc-600m`, `boltz2`. The family is a naming convention, not a mechanism: a variant's directory is a copy of its sibling's with the weights line and the skill's numbers changed. Containerfiles put the environment first and the weights last, so variants share their base layers in the registry and on the nodes.
- **Weights are in the image**, fetched at build time, never at run time (`HF_HUB_OFFLINE=1` in the image). A version of a tool is one immutable thing. The ceiling is databases too big for an image (AlphaFold's); those belong to the data companion, not the shed.
- **Every example in a skill is a full command line**, the program on the image's PATH with its arguments, never a bare argument to an entrypoint: pyxis ignores entrypoints and runs the job's command inside the container.
- **The image carries its own test.** The Containerfile copies the tool's `test/` directory to `/toolshed/test/`, so `bash /toolshed/test/run.sh` checks the image wherever it runs. The fixture is kilobytes.

**The skill** is the tool's whole description, prose in the Agent Skills format Claude Code, Codex, and OpenCode read:

```markdown
---
name: esm2-650m
description: ESM-2 650M protein language model: per-residue embeddings, pseudo-log-likelihoods, and masked-position infilling for protein sequences. Reach for it for a sequence score or an embedding when a structure is not needed.
metadata:
  version: 1.0.0
---
# esm2-650m
Image: `ghcr.io/brineylab/toolshed/esm2-650m:1.0.0`
## Commands
## Inputs and outputs
## Compute
## Running it
## Gotchas
```

The frontmatter has three fields a machine reads: `name` (the directory's name), `description` (the catalogue line), and `metadata.version` (the image tag). Everything else is prose for the agent. No declared parameter lists; if agents are seen misusing commands from prose, that is the trigger for structure.

## 2. The repository

Public, one directory per tool:

```
README.md                 what the shed is; finding and running a tool from anywhere; installing the skills
CONTRIBUTING.md           adding a tool: the layout, the conventions, the test, versions, the review
docs/                     this design, the vision, the plans, the builder's setup
tools/<name>/SKILL.md     the skill (§1)
tools/<name>/Containerfile
tools/<name>/test/        run.sh and a tiny fixture: exits 0 iff the tool works
tools/<name>/bin/         the wrapper program, where upstream has none
scripts/check.py          a tool directory's rules, and whether its version is already published
scripts/build <name> <image>   podman build
scripts/test <image>      the image's test with podman on this machine
.github/workflows/tools.yml
```

**Versions.** `metadata.version` is the image tag, `ghcr.io/<owner>/<repo>/<name>:<version>`, with the owner and repository from the shed's own GitHub path, so a fork is its own shed and registry with no edits. A published tag is never overwritten: any change under `tools/<name>/` bumps the version, prose included, and CI refuses a pull request whose version is already in the registry. Callers always name a version, never a floating tag.

**The registry** is GHCR. Packages are public: free in storage and transfer, and pulled by enroot on cluster nodes and podman on laptops with no credential. The organization's package default is set to public once. A tool whose license forbids redistribution is the exception, handled when it arrives (likely a squashfs on the cluster's shared storage), not designed for now.

**The test** is the documented commands on a tiny fixture with a known answer, run inside the built image: it checks the environment, the weights, and that the skill's examples are true, in one step. It must pass on a cluster under pyxis, because that is where an image must work and where it differs from `podman run`: entrypoints ignored, home mounted, the job's environment propagated, the node's driver. So the test is not CI's: CI has no GPU and no cluster. Whoever asks for the tool runs it where they will use it. A flyloft campaign runs it as one of its runs (§6); a person runs `srun --container-image=... bash /toolshed/test/run.sh` on their cluster, or the `podman run` form on a laptop, and pastes the output into the pull request. One GPU and 30 minutes is the expectation; the test passes with no network.

**The builder** is a self-hosted GitHub Actions runner on a lab machine, as its own OS user with podman and nothing else: no cluster key, no other service's configuration, no tokens but the workflow's own. Builds go there because images with weights are 5 to 25 GB, beyond a hosted runner's disk. A cluster is never asked to build: rootless podman on compute nodes is the one thing this design does not require of it, and enroot cannot build at all.

**The workflow**, `tools.yml`, with the tools a pull request changes as its unit:

- `check`, on a hosted runner, on every pull request: the skill's frontmatter parses and its name is the directory's; a Containerfile and `test/run.sh` exist; the version is not in the registry.
- `build`, on the self-hosted runner, only for a pull request whose branch is in this repository (`github.event.pull_request.head.repo.full_name == github.repository`): build, then push as `<name>:pr-<number>`, the candidate. A candidate is pullable like any published image, so the test can run against it anywhere.
- `promote`, on a push to `main`, for the tools the push changed: the candidate tag copied to the version tag (`skopeo copy` within the registry writes a manifest and moves no layers). It runs on the push, not on the pull request's close, so that its check attaches to the merge commit, which is what flyloft watches.

**Guards.** Only organization members can push branches into the repository, so only their pull requests build; a fork's pull request gets `check` and nothing runs on the lab's machine. The repository setting that requires approval for all outside collaborators' workflows is a second belt. A lab member who wants an outsider's tool pushes the branch into the repository. The runner builds whatever a member's Containerfile says, as its own user, with the disk and network that implies and nothing more.

**From a laptop.** The skill's `podman run` line is the interface. To give an agent the shed's skills: clone the repository and symlink each `tools/<name>` into the harness's skills directory (the README has the line). No `toolshed` command until the prose proves insufficient.

**Weights behind a login** (Hugging Face gated repositories): the Containerfile takes the token as a build secret from the repository's secrets.

## 3. Review

Every tool is reviewed by someone other than its builder before it is published: a lab member for a person's pull request, a flyloft critic for a campaign's. The reviewer asks: does the skill describe what the Containerfile builds, and is every command it documents exercised by the test; are the weights fetched at build time, so the image is self-contained; is every example a full command line; is the version new and the name the directory's; did the test pass where it was run, and does its output show the commands ran. The reviewer never asks for more than a working, truthfully documented tool.

## 4. The first tools

`hello` (checks that a machine or cluster runs shed images), `esm2-650m`, and `esmc-600m`, built by hand. `boltz2` is the first tool a flyloft campaign builds. Sizes beyond these are added when a question asks for one.

## 5. Tests

The shed has no test tier of its own beyond `scripts/check.py`'s unit test: it is a repository, a workflow, and tools. Its acceptance is live, in [plans/toolshed-a.md](plans/toolshed-a.md): a pull request builds on the lab's runner and pushes its candidate; the candidate's test passes on the cluster under pyxis; merge promotes the tag; the published image runs from a laptop with the skill's line; a changed tool without a version bump is refused; a fork's pull request runs nothing on the lab's machine.

## 6. How flyloft uses the shed

Designed in flyloft's `docs/design/toolshed.md`. In short: flyloft keeps a clone at `origin/main` and renders the catalogue (name, version, description, skill path) into its implementers' assignments; a run may name `tool="name@version"`, and its job then runs inside the tool's image with the work directory mounted; an implementer that needs a tool the shed lacks writes it in a worktree of the shed and submits it; flyloft pushes the branch, opens the pull request, waits for `build`, runs `bash tools/<name>/test/run.sh` inside the candidate image as a run of the campaign on the campaign's cluster and budget, has its critic review the pull request and the test run, merges with a merge commit, and watches the merge commit's `promote` check; then the campaign calls the published version. One tool build at a time per campaign.

## 7. Deferred

| Item | Picked up |
|---|---|
| A private tool (a license that forbids redistribution): a squashfs on the cluster's shared storage | When one arrives |
| A `toolshed` command for laptops that picks podman or docker and mounts the working directory | When the skill's `podman run` line proves insufficient |
| Declared parameters per command | When agents are seen misusing commands from prose |
| Bulk databases (AlphaFold's) | The data companion's design |
| An image per GPU architecture (Arm) | When an Arm partition is wanted |
| A tool whose test needs more than one GPU or 30 minutes | When one is built |
| Retiring tools nobody calls: a count of callers | When the shed is big enough to need one |
| Running the test on CPU in CI where a tool allows it | When a broken candidate reaches a cluster often enough to matter |
