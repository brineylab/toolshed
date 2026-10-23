# Toolshed, Plan A: The Shed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The lab's toolshed exists as a public repository, `brineylab/toolshed`, with its check script, its workflow (check, build on the lab's runner, promote), the builder registered on the flyloft host, and three tools published to GHCR: `hello`, `esm2-650m`, and `esmc-600m`, each tested on slinky under pyxis.

**Architecture:** One directory per tool holding a prose skill, a Containerfile, a test the image carries at `/toolshed/test`, and a wrapper program where upstream has none. CI never runs a tool's test: the `check` job validates the directory and the version on a hosted runner, the `build` job builds and pushes a candidate tag `pr-<n>` on the self-hosted runner for branches in this repository only, and `promote` copies the candidate to the version tag on a push to `main`. Whoever asks for a tool tests the candidate where they will use it: a person with `srun` or `podman`; flyloft as a run (plan B). The scripts are stdlib Python and bash; nothing in the repository names the lab.

**Tech Stack:** podman (rootless), GHCR, GitHub Actions with one self-hosted runner, skopeo, Python 3.11 stdlib, bash. Tools: PyTorch CUDA base images, `transformers` and `huggingface_hub` (ESM-2), the `esm` package (ESM-C).

**Spec:** `docs/DESIGN.md` (this repository); the flyloft side is flyloft's `docs/design/toolshed.md`.

## Global Constraints

- The repository is `brineylab/toolshed`, public, default branch `main`. Nothing in `scripts/` or `.github/` names the lab: the registry path is `ghcr.io/${GITHUB_REPOSITORY,,}/<name>`, the candidate tag `pr-<pull request number>`, the version tag `metadata.version`.
- A tool directory is `tools/<name>/` with `SKILL.md`, `Containerfile`, `test/run.sh`, optionally `bin/` and more files under `test/`. `<name>` matches `^[a-z0-9]+(-[a-z0-9]+)*$`; the skill's `name` equals it; `metadata.version` matches `^\d+\.\d+\.\d+$`.
- Every Containerfile ends with `COPY test /toolshed/test` after its weights layer, sets the environment first and fetches weights last, and sets `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1` where transformers is used) so nothing downloads at run time. Pinned versions for every package and base image.
- A tool's test is `bash /toolshed/test/run.sh` inside the image: exit 0 iff the tool works; it writes only under `/tmp`. It passes with `--network none` under podman.
- A published tag is never overwritten: `check` refuses a changed tool whose `metadata.version` is already a tag in the registry; `promote` refuses the same.
- Scripts are Python 3.11 stdlib (`python3 -I`) or bash with `set -euo pipefail`; no third-party package in the repository itself.
- Skills are in the Agent Skills format: frontmatter `name`, `description`, `metadata: {version}`; every example is a full command line with the program on PATH, never an entrypoint argument.
- Commit messages: one sentence stating the behavior; end with the session's `Co-Authored-By` trailer. Each task ends in a commit; tasks 3 to 6 end in a merged pull request on the shed.

## Review Focus

1. **A pull request that changes a tool's prose only, with the same version:** `check` refuses it with `tools/<name>: version 1.0.0 is already published; bump metadata.version`. Test in Task 1 (`test_check.py`, with the registry stubbed) and live in Task 4.
2. **A pull request from a fork:** `build` does not run on the host; `check` still runs. Verified in Task 3 with a fork of the repository.
3. **An image on a node with no internet:** `scripts/test` runs with `--network none` and the test passes; a Containerfile that fetches weights at run time fails this. Test in Task 4 and Task 5.
4. **The image under pyxis:** no entrypoint; `bash /toolshed/test/run.sh` finds its files; `$HOME` is the node user's, not the image's. Verified by the `srun` step in Tasks 3, 4, and 5.
5. **A candidate that was never promoted** (a pull request closed without merging): `pr-<n>` stays in the registry as a dangling tag. Accepted; Task 2's workflow comment says so, and a dangling candidate is deleted by hand from the package's page if it matters.

---

### Task 1: The repository, the check script, and its test

**Files:**
- Create: `README.md`, `CONTRIBUTING.md`, `scripts/check.py`, `scripts/build`, `scripts/test`, `tests/test_check.py`, `.github/pull_request_template.md`, `.gitignore`
- Modify: `AGENTS.md` (written before this plan, with `CLAUDE.md` importing it)

**Interfaces:**
- Produces: `scripts/check.py <tool dir>... [--registry ghcr.io/<owner>/<repo>]`: exit 0 when every directory passes, else prints one `tools/<name>: <reason>` line per failure and exits 1. Importable: `frontmatter(path: Path) -> dict` (the parsed block, `metadata` as a nested dict), `check_tool(tool_dir: Path, published: set[str] | None) -> list[str]` (the reasons), `published_tags(registry: str, name: str) -> set[str]` (GHCR's tags for the package, `set()` when the package does not exist).
- Produces: `scripts/build <name> <image>`: `podman build --secret id=hf_token,env=HF_TOKEN -t <image> -f tools/<name>/Containerfile tools/<name>` (the secret only when `HF_TOKEN` is set).
- Produces: `scripts/test <image>`: `podman run --rm --network none [--device nvidia.com/gpu=all when nvidia-smi is on PATH] <image> bash /toolshed/test/run.sh`.

- [ ] **Step 1: Create the repository**

On GitHub, as the organization: `brineylab/toolshed`, public, empty (no README, no license yet). Locally: `git init -b main`, add the remote.

- [ ] **Step 2: Write the failing test**

`tests/test_check.py` (stdlib `unittest`; it builds tool directories under `tempfile`):

```python
import unittest, tempfile, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check  # noqa: E402

SKILL = "---\nname: {name}\ndescription: A tool.\nmetadata:\n  version: {version}\n---\n# {name}\n"

def tool(root: Path, name="esm2-650m", version="1.0.0", skill=True, containerfile="FROM x\nCOPY test /toolshed/test\n", run=True):
    d = root / "tools" / name
    (d / "test").mkdir(parents=True)
    if skill: (d / "SKILL.md").write_text(SKILL.format(name=name, version=version))
    if containerfile is not None: (d / "Containerfile").write_text(containerfile)
    if run: (d / "test" / "run.sh").write_text("set -euo pipefail\necho ok\n")
    return d

class CheckTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
    def test_frontmatter_parses_name_description_and_nested_version(self):
        d = tool(self.root)
        self.assertEqual(check.frontmatter(d / "SKILL.md"), {"name": "esm2-650m", "description": "A tool.", "metadata": {"version": "1.0.0"}})
    def test_a_good_tool_passes(self):
        self.assertEqual(check.check_tool(tool(self.root), published=set()), [])
    def test_every_rule_names_its_reason(self):
        cases = [
            (dict(name="Esm2"), "name"),                       # directory name not lowercase-hyphen
            (dict(version="1.0"), "metadata.version"),          # not semver
            (dict(skill=False), "SKILL.md"),
            (dict(containerfile=None), "Containerfile"),
            (dict(containerfile="FROM x\n"), "COPY test /toolshed/test"),
            (dict(run=False), "test/run.sh"),
        ]
        for kw, word in cases:
            root = Path(tempfile.mkdtemp())
            reasons = check.check_tool(tool(root, **kw), published=set())
            self.assertTrue(any(word in r for r in reasons), (kw, reasons))
    def test_a_skill_whose_name_is_not_the_directorys_fails(self):
        d = tool(self.root)
        (d / "SKILL.md").write_text(SKILL.format(name="other", version="1.0.0"))
        self.assertTrue(any("name" in r for r in check.check_tool(d, published=set())))
    def test_a_published_version_is_refused(self):
        reasons = check.check_tool(tool(self.root), published={"1.0.0"})
        self.assertEqual(reasons, ["version 1.0.0 is already published; bump metadata.version"])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python3 -I -m unittest tests/test_check.py -v`
Expected: FAIL (`No module named 'check'`).

- [ ] **Step 4: Implement `scripts/check.py`**

Stdlib only. `frontmatter` reads the first `---` block: a line `key: value` is a top-level string; a line `key:` with nothing after it opens a nested dict of the following two-space-indented `key: value` lines. `check_tool` returns the reasons in this order: directory name, `SKILL.md` present, `name` equals the directory, `description` non-empty, `metadata.version` semver, `Containerfile` present, `COPY test /toolshed/test` in it, `test/run.sh` present, then `version <v> is already published; bump metadata.version` when `published` holds it. `published_tags` exchanges an anonymous token (`GET https://ghcr.io/token?scope=repository:<owner/repo>/<name>:pull`, read `token`) and lists `GET https://ghcr.io/v2/<owner/repo>/<name>/tags/list` with `Authorization: Bearer`; a 404 is `set()`. `main`: each argument a tool directory; with `--registry`, `published` comes from `published_tags`, else `None` (the version rule is skipped); prints `tools/<name>: <reason>` lines; exit 1 on any.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -I -m unittest tests/test_check.py -v`
Expected: PASS, six tests.

- [ ] **Step 6: Write `scripts/build`, `scripts/test`, the README, CONTRIBUTING, the pull request template**

`scripts/build` and `scripts/test` as the Interfaces block says; both `#!/usr/bin/env bash` with `set -euo pipefail`, executable.

`README.md`: what the shed is (two paragraphs from the design's opening and §2); *Finding a tool*: `ls tools`, each `tools/<name>/SKILL.md` starts with what it is for; *Running a tool from a laptop*: the `podman run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work ghcr.io/brineylab/toolshed/<name>:<version> <command>` line (the GPU device needs the NVIDIA container toolkit's CDI spec once: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`; without a GPU, drop `--device`), with `docker` where that is what exists; *On a Slurm cluster with pyxis*: `--container-image=ghcr.io#brineylab/toolshed/<name>:<version>` on `srun` or `#SBATCH`, with `--container-mounts` for the directories the command reads and writes; *Giving your coding agent the skills*: `git clone`, then `for t in tools/*; do ln -s "$PWD/$t" ~/.claude/skills/; done` (and `~/.agents/skills/` for Codex and OpenCode); *Checking an image*: `bash /toolshed/test/run.sh` inside it; *Versions*: immutable tags, always name one.

`CONTRIBUTING.md`: *Adding a tool*: copy the nearest sibling's directory; the four files and what each holds; the conventions (environment first, weights last, `HF_HUB_OFFLINE=1`, `COPY test /toolshed/test` last, pinned versions, one image per size variant, a wrapper program only where upstream has none, every skill example a full command line); *The skill*: the §2 outline (Image, Commands, Inputs and outputs, Compute, Running it, Gotchas) and the three frontmatter fields; *The test*: `test/run.sh` on a tiny fixture, exit 0 iff the tool works, writes under `/tmp` only, must pass with no network; *Building and testing locally*: `scripts/build <name> ghcr.io/brineylab/toolshed/<name>:dev` then `scripts/test ghcr.io/brineylab/toolshed/<name>:dev`; *Versions*: bump `metadata.version` on any change, prose included; *The pull request*: push a branch in this repository (a fork's pull request does not build), CI builds `pr-<n>`, test the candidate where you will use it, `srun --partition=<partition> --gres=gpu:1 --time=00:30:00 --container-image=ghcr.io#brineylab/toolshed/<name>:pr-<n> bash /toolshed/test/run.sh`, paste the output into the pull request, a reviewer merges, `promote` publishes the version; *Weights behind a login*: `RUN --mount=type=secret,id=hf_token HF_TOKEN=$(cat /run/secrets/hf_token) ...` in the Containerfile, the repository secret `HF_TOKEN` on the runner.

`.github/pull_request_template.md`: `## Tool`, `## Test output` (where it was run, pasted output), `## Checklist` (version bumped; skill examples run; test passes with no network).

`.gitignore`: `__pycache__/`.

`AGENTS.md`: drop "once it exists" from its pointer to `CONTRIBUTING.md`. Its rules stay short and binding; the step-by-step lives in CONTRIBUTING and is not repeated in AGENTS.md.

- [ ] **Step 7: Commit and push**

```bash
git add -A
git commit -m "The toolshed: one directory per tool, a check script with its test, build and test scripts, and the guidance for adding a tool"
git push -u origin main
```

---

### Task 2: The workflow

**Files:**
- Create: `.github/workflows/tools.yml`
- Modify: `scripts/check.py` (nothing; used), `README.md` (one line under *Versions*: what CI does)

**Interfaces:**
- Consumes: `scripts/check.py`, `scripts/build` from Task 1.
- Produces: the jobs `check`, `build`, `promote`; the candidate tag `ghcr.io/<repo>/<name>:pr-<n>`; the repository variable none; the repository secret `HF_TOKEN` (optional).

- [ ] **Step 1: Write the workflow**

`on`: `pull_request` with `paths: ["tools/**"]`; `push` to `main` with `paths: ["tools/**"]`. `permissions: {contents: read, packages: write, pull-requests: read}`. `env: REGISTRY: ghcr.io`.

A reusable first step in every job computes the changed tools into `$GITHUB_OUTPUT` as `tools` (space-separated names): on `pull_request`, `git diff --name-only origin/${{ github.base_ref }}...HEAD -- tools/ | cut -d/ -f2 | sort -u` after `git fetch origin ${{ github.base_ref }}`; on `push`, `git diff --name-only ${{ github.event.before }} ${{ github.sha }} -- tools/ | cut -d/ -f2 | sort -u` (checkout with `fetch-depth: 0`).

Jobs:

- `check` (`runs-on: ubuntu-latest`, `if: github.event_name == 'pull_request'`): checkout; `python3 -I -m unittest discover tests`; `python3 -I scripts/check.py $(for t in $TOOLS; do echo tools/$t; done) --registry "$REGISTRY/${GITHUB_REPOSITORY,,}"`.
- `build` (`runs-on: [self-hosted, builder]`, `needs: check`, `if: github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name == github.repository`): checkout; `podman login $REGISTRY -u ${{ github.actor }} --password-stdin <<< "${{ secrets.GITHUB_TOKEN }}"`; for each tool: `IMAGE=$REGISTRY/${GITHUB_REPOSITORY,,}/$t:pr-${{ github.event.number }}`, `HF_TOKEN=${{ secrets.HF_TOKEN }} scripts/build $t $IMAGE`, `podman push $IMAGE`, then `echo "candidate: $IMAGE"` into `$GITHUB_STEP_SUMMARY`; last step always: `podman image prune -f --filter until=168h`. A comment above the job: the branch guard and why; a candidate of a pull request closed unmerged stays in the registry (Review Focus 5).
- `promote` (`runs-on: ubuntu-latest`, `if: github.event_name == 'push'`): checkout; `sudo apt-get install -y skopeo`; `skopeo login $REGISTRY -u ${{ github.actor }} -p ${{ secrets.GITHUB_TOKEN }}`; `NUMBER=$(gh api repos/$GITHUB_REPOSITORY/commits/$GITHUB_SHA/pulls --jq '.[0].number')` with `GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}`; for each tool: `VERSION=$(python3 -I -c "import sys; sys.path.insert(0,'scripts'); import check, pathlib; print(check.frontmatter(pathlib.Path('tools/$t/SKILL.md'))['metadata']['version'])")`; refuse if `skopeo inspect docker://$REGISTRY/${GITHUB_REPOSITORY,,}/$t:$VERSION` succeeds (`echo "$t:$VERSION already published" >&2; exit 1`); `skopeo copy docker://...:pr-$NUMBER docker://...:$VERSION`.

- [ ] **Step 2: Lint the workflow**

Run: `podman run --rm -v "$PWD":/repo:ro -w /repo docker.io/rhysd/actionlint:latest -color`
Expected: no findings.

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/tools.yml README.md
git commit -m "CI checks a tool's directory and version, builds a candidate on the lab's runner for branches in this repository, and promotes it to the version tag on merge"
git push
```

---

### Task 3: The builder on the host, the settings, and the `hello` tool

**Files:**
- Create: `tools/hello/SKILL.md`, `tools/hello/Containerfile`, `tools/hello/test/run.sh`, `docs/builder.md`

**Interfaces:**
- Produces: a runner with labels `self-hosted, builder` in the organization's runner group, serving `toolshed` (and later `toolshed-dev`); the published image `ghcr.io/brineylab/toolshed/hello:1.0.0`.

- [ ] **Step 1 [human]: Organization and repository settings**

Organization → Packages: *Package creation* default visibility **public**. Organization → Actions → Runner groups: the default group (or a new `builders` group) with *Allow public repositories* on. Repository → Actions → General: *Fork pull request workflows from outside collaborators*: **Require approval for all outside collaborators**. Repository → Secrets: `HF_TOKEN` (a Hugging Face read token of a lab account that has accepted the ESM-C license), to be used by Task 5.

- [ ] **Step 2 [human, on the flyloft host]: The runner user**

As root: `useradd -m toolshed`; `usermod --add-subuids 200000-265535 --add-subgids 200000-265535 toolshed`; `loginctl enable-linger toolshed`; confirm `podman` is installed. As `toolshed` (`machinectl shell toolshed@` or `sudo -iu toolshed`): `podman info` runs; `podman run --rm docker.io/library/alpine:3.20 echo ok` prints `ok`. Disk: `df -h ~` shows at least 200 GB free; if not, point `~/.config/containers/storage.conf` `graphroot` at a volume that has it.

- [ ] **Step 3 [human, as `toolshed`]: Register the runner**

Organization → Actions → Runners → New runner → Linux x64: follow the download and `./config.sh --url https://github.com/brineylab --token <token> --labels builder --unattended` steps in `~/actions-runner`; then `sudo ./svc.sh install toolshed && sudo ./svc.sh start` (root, from the same directory), and confirm the runner shows *Idle* on GitHub with labels `self-hosted`, `Linux`, `X64`, `builder`.

- [ ] **Step 4: Write `docs/builder.md`**

Steps 1 to 3 as prose a lab operator follows, plus: how to see the runner's logs (`journalctl --user -u actions.runner.*` as `toolshed`), where images are cached (`~/.local/share/containers`), and the one-line prune the workflow runs.

- [ ] **Step 5: Write the `hello` tool**

`tools/hello/SKILL.md`: `name: hello`, `description: "Checks that this machine or cluster runs toolshed images: prints the OS, the user, and the GPU it sees. Reach for it first on a new cluster, before any real tool."`, `metadata: {version: 1.0.0}`; body: Image, Commands (`hello` prints three lines: `os=<id>`, `user=<uid>`, `gpu=<nvidia-smi name or none>`), Running it (the laptop line and the `srun --container-image=ghcr.io#brineylab/toolshed/hello:1.0.0 hello` line), Gotchas (under pyxis the user is the node user, not root).

`tools/hello/Containerfile`: `FROM docker.io/library/alpine:3.20`; `COPY bin/hello /usr/local/bin/hello`; `RUN chmod +x /usr/local/bin/hello`; `COPY test /toolshed/test`. `tools/hello/bin/hello`: a `sh` script printing the three lines (`gpu=` from `nvidia-smi --query-gpu=name --format=csv,noheader` when present, else `none`). `tools/hello/test/run.sh`: `set -eu; out=$(hello); echo "$out"; case "$out" in *"os="*"user="*"gpu="*) ;; *) exit 1;; esac`.

- [ ] **Step 6: Test locally**

Run: `scripts/build hello ghcr.io/brineylab/toolshed/hello:dev && scripts/test ghcr.io/brineylab/toolshed/hello:dev`
Expected: three lines and exit 0.

- [ ] **Step 7: Open the pull request and watch CI**

```bash
git checkout -b hello
git add tools/hello docs/builder.md
git commit -m "The hello tool checks that a machine or cluster runs toolshed images, and the builder's setup is documented"
git push -u origin hello
gh pr create --fill
```

Expected on the pull request: `check` passes; `build` runs on the host's runner and its summary names `ghcr.io/brineylab/toolshed/hello:pr-1`.

- [ ] **Step 8 [human]: Test the candidate on slinky and paste the output**

Run on slinky's login node: `srun --partition=<a GPU partition from clusters/slinky.toml> --gres=gpu:1 --time=00:10:00 --container-image=ghcr.io#brineylab/toolshed/hello:pr-1 bash /toolshed/test/run.sh`
Expected: `os=alpine`, `user=<your uid>`, a GPU name, exit 0. Paste into the pull request. This is the first image pulled from GHCR by enroot on slinky: if the pull fails, note what enroot says (credentials, cache path) in the pull request before anything else is tried.

- [ ] **Step 9: Merge and confirm promotion**

`gh pr merge --merge --delete-branch`. Expected: `promote` runs on the push and `skopeo inspect docker://ghcr.io/brineylab/toolshed/hello:1.0.0` succeeds; the package is public (an anonymous `podman pull ghcr.io/brineylab/toolshed/hello:1.0.0` works from a laptop with no login).

- [ ] **Step 10: Verify the fork guard**

From a personal fork, open a pull request that touches `tools/hello/SKILL.md`. Expected: `check` runs (or waits for approval, per the repository setting); `build` shows as skipped. Close it unmerged.

---

### Task 4: `esm2-650m`

**Files:**
- Create: `tools/esm2-650m/SKILL.md`, `tools/esm2-650m/Containerfile`, `tools/esm2-650m/bin/esm2`, `tools/esm2-650m/test/run.sh`, `tools/esm2-650m/test/seqs.fasta`, `tools/esm2-650m/test/masked.fasta`, `tools/esm2-650m/test/check.py`

**Interfaces:**
- Produces: `esm2 pll --in <fasta> --out <tsv>` (columns `id`, `pll`: the pseudo-log-likelihood, each position masked in turn, summed), `esm2 embed --in <fasta> --out <npz>` (`ids`: array of str, `embeddings`: float32 of shape (n, 1280), the last hidden layer mean-pooled over residues), `esm2 fill --in <fasta> --out <fasta>` (every `<mask>` token replaced by the argmax residue). One sequence at a time; CUDA when available, else CPU; `--device` to force one.

- [ ] **Step 1: Write the test**

`test/seqs.fasta`: two sequences of 12 to 20 residues (a peptide and its reverse). `test/masked.fasta`: one of them with two positions as `<mask>`. `test/check.py <pll.tsv> <emb.npz> <filled.fasta>`: the TSV has two rows with finite values below 0; the npz has `embeddings.shape == (2, 1280)` and `ids` of length 2; the filled FASTA has no `<mask>` and the same length as its input. `test/run.sh`:

```bash
set -euo pipefail
cd "$(dirname "$0")"
esm2 pll --in seqs.fasta --out /tmp/pll.tsv
esm2 embed --in seqs.fasta --out /tmp/emb.npz
esm2 fill --in masked.fasta --out /tmp/filled.fasta
python3 check.py /tmp/pll.tsv /tmp/emb.npz /tmp/filled.fasta
```

- [ ] **Step 2: Write the Containerfile**

`FROM docker.io/pytorch/pytorch:<current 2.x>-cuda12.<x>-cudnn9-runtime`; `RUN pip install --no-cache-dir transformers==<current> huggingface_hub==<current> numpy==<current>`; `ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; `COPY bin/esm2 /usr/local/bin/esm2` and `chmod +x`; `RUN HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esm2_t33_650M_UR50D', allow_patterns=['*.json', '*.safetensors', 'vocab.txt'])"`; `COPY test /toolshed/test`. Pin the exact versions current at build time.

- [ ] **Step 3: Write `bin/esm2`**

Python, `argparse` with the three subcommands; `AutoTokenizer` and `EsmForMaskedLM` from `transformers` loaded from the offline cache; FASTA read with a ten-line parser (no Biopython). `pll`: for each position i, mask it, forward, add the log-softmax at i of the true residue. `embed`: `output_hidden_states=True`, last layer, mean over tokens 1..L. `fill`: forward once, argmax at mask positions, decode.

- [ ] **Step 4: Build and test locally**

Run: `scripts/build esm2-650m ghcr.io/brineylab/toolshed/esm2-650m:dev && scripts/test ghcr.io/brineylab/toolshed/esm2-650m:dev`
Expected: exit 0 on a laptop without a GPU (CPU, under a minute) and with one. Then: `podman run --rm --network none ghcr.io/brineylab/toolshed/esm2-650m:dev esm2 pll --in /toolshed/test/seqs.fasta --out /tmp/x.tsv` exits 0 (no download attempted).

- [ ] **Step 5: Write the skill**

`SKILL.md` per the design's §2 outline: frontmatter (`name: esm2-650m`, a description that says what it is for and when to reach for it, `metadata: {version: 1.0.0}`); Image; Commands (the three, each as a full line with its inputs and outputs); Inputs and outputs (FASTA in, TSV / npz / FASTA out, what `pll` means and its sign); Compute (one GPU, memory, and a measured rate from Step 4: sequences per second for `pll` at a stated length); Running it (the laptop line from the README with a real command; the flyloft line `request_run(tool="esm2-650m@1.0.0", command="esm2 pll --in data/seqs.fasta --out $FLYLOFT_OUT/pll.tsv", gpus=1, time_limit="00:30:00")`); Gotchas (`<mask>` is the mask token; sequences longer than 1022 residues are refused by the model; CPU is fine for a few sequences).

- [ ] **Step 6: Pull request, candidate, cluster test, merge**

Branch `esm2-650m`; commit `ESM-2 650M as a tool: pseudo-log-likelihoods, embeddings, and infilling in one image`; push; `gh pr create --fill`. When `build` finishes: on slinky, `srun --partition=<p> --gres=gpu:1 --time=00:30:00 --container-image=ghcr.io#brineylab/toolshed/esm2-650m:pr-<n> bash /toolshed/test/run.sh`, expected exit 0 and `gpu` in use (`nvidia-smi` shows the process, or `pll` finishes in seconds). Paste the output. Merge with `--merge --delete-branch`; expected `promote` publishes `1.0.0`.

- [ ] **Step 7: Verify the version rule live (Review Focus 1)**

Branch `esm2-prose`: edit one sentence in `tools/esm2-650m/SKILL.md`, no version change; push; open the pull request. Expected: `check` fails with `tools/esm2-650m: version 1.0.0 is already published; bump metadata.version`. Close it.

- [ ] **Step 8: Run it from a laptop**

With a `seqs.fasta` in the working directory, the skill's laptop line. Expected: `pll.tsv` written.

---

### Task 5: `esmc-600m`

**Files:**
- Create: `tools/esmc-600m/` with the same four parts as Task 4 (`bin/esmc`)

**Interfaces:**
- Produces: `esmc pll`, `esmc embed` (shape (n, 1152)), `esmc fill`, with the same arguments as `esm2`.

- [ ] **Step 1: Copy the sibling and change what differs**

`cp -r tools/esm2-650m tools/esmc-600m`; rename `bin/esm2` to `bin/esmc`; `test/check.py` expects 1152; `run.sh` calls `esmc`.

- [ ] **Step 2: Write the Containerfile**

Same base; `pip install --no-cache-dir esm==<current>` (EvolutionaryScale's package) and its pins; weights: `RUN --mount=type=secret,id=hf_token HF_TOKEN=$(cat /run/secrets/hf_token) HF_HUB_OFFLINE=0 python -c "from esm.models.esmc import ESMC; ESMC.from_pretrained('esmc_600m')"` if the weights are gated (the first build tells: a 401 from Hugging Face), else without the secret; `ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1`; `COPY test /toolshed/test` last.

- [ ] **Step 3: Write `bin/esmc`**

The `esm` package's API: `ESMC.from_pretrained("esmc_600m").to(device)`, `ESMProtein(sequence=...)`, `model.encode`, `model.logits(..., LogitsConfig(sequence=True, return_embeddings=True))`. `pll` masks with the package's mask character (`_`) one position at a time; `embed` mean-pools the returned embeddings over residues; `fill` replaces `_` (the skill says to write `_` for a masked position in this tool).

- [ ] **Step 4: Build and test locally**

As Task 4 Step 4, with `HF_TOKEN` exported if gated. Expected: exit 0 with `--network none`.

- [ ] **Step 5: Write the skill**

As Task 4 Step 5, with the ESM-C facts: 1152-dimensional embeddings, `_` as the mask, the license is non-commercial (one line under Gotchas), and a measured rate.

- [ ] **Step 6: Pull request, candidate, cluster test, merge**

As Task 4 Step 6, branch `esmc-600m`, commit `ESM-C 600M as a tool: pseudo-log-likelihoods, embeddings, and infilling in one image`. If `build` fails with a Hugging Face 401, add the `HF_TOKEN` secret (Task 3 Step 1) and the `--mount=type=secret` line, push, and let it rebuild; record which it was in the pull request.

---

### Task 6: The scratch shed for flyloft's tier two

**Files:** none in this repository.

- [ ] **Step 1 [human]: Create `brineylab/toolshed-dev`**

A public repository created from the shed at `main` after Task 5 (`gh repo create brineylab/toolshed-dev --public --clone` then push the shed's `main` into it), with the same Actions settings as Task 3 Step 1 and served by the same runner group. Its registry is `ghcr.io/brineylab/toolshed-dev/`.

- [ ] **Step 2: Confirm the workflow runs there**

Open a pull request in `toolshed-dev` bumping `tools/hello` to `1.0.1`; expected: `check` passes, `build` runs on the host and pushes `ghcr.io/brineylab/toolshed-dev/hello:pr-1`; merge; expected `promote` publishes `1.0.1`. `toolshed-dev` is plan B's tier-two shed (`docs/design/toolshed.md` §6); it is not a place for real tools.
