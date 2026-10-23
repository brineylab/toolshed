# Toolshed, Plan A: The Shed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The lab's toolshed exists as a public repository, `brineylab/toolshed`, with its check script, the contributor's build, push, and test scripts, a workflow that checks a pull request's candidate and record and promotes on merge, and three tools published to GHCR: `hello`, `esm2-650m`, and `esmc-600m`, each tested on slinky under pyxis.

**Architecture:** One directory per tool holding a prose skill, a Containerfile, a test the image carries at `/toolshed/test`, and a wrapper program where upstream has none. The contributor builds the image for `linux/amd64`, pushes it as the candidate `<name>:src-<tree>` (git's hash of `tools/<name>/`), runs its test by digest where the tool will be used, and commits the record `results/<name>/<version>.txt`. CI, on hosted runners only, never builds, pulls, or runs an image: `check` validates the directory, the version, the candidate (present, public, amd64), and the record (its digest is the candidate's, exit 0); `promote` copies the candidate to the version tag on a push to `main`. The scripts are stdlib Python and bash; nothing in the repository names the lab.

**Tech Stack:** podman (rootless) on the contributor's machine, GHCR, GitHub Actions on hosted runners, skopeo, Python 3 stdlib, bash, Slurm with pyxis and enroot. Tools: PyTorch CUDA base images, `transformers` and `huggingface_hub` (ESM-2), the `esm` package (ESM-C).

**Spec:** `docs/DESIGN.md` (this repository), amended 2026-10-23: contributors build and test candidates, CI never builds or runs one. The flyloft side is flyloft's `docs/design/toolshed.md`, whose §6 summary in DESIGN changed with the amendment.

**Revision, 2026-10-23.** Task 1 was executed and pushed against the original design. The original Task 2 (a workflow with a `build` job on a self-hosted runner) was executed and reviewed locally and not pushed (commits `359048c`, `511f9dd` on `main`); the amendment removed that runner. Its `scripts/changed-tools`, `scripts/promote`, their tests, and the workflow's `check` and `promote` jobs carry over; Task 4 below removes the `build` job and changes `promote`'s candidate. The original Tasks 3 to 6 are replaced by Tasks 5 to 8.

## Global Constraints

- The repository is `brineylab/toolshed`, public, default branch `main`. Nothing in `scripts/` or `.github/` names the lab: the registry is `ghcr.io/<owner>/<repo>` lowercased, from `GITHUB_REPOSITORY` in CI and from `git remote get-url origin` locally.
- The candidate tag is `src-<tree>` with `<tree>` = `git rev-parse HEAD:tools/<name>` (40 hex characters); the version tag is `metadata.version`. A published version tag is never overwritten.
- Every image is `linux/amd64` and carries the label `org.opencontainers.image.source=https://github.com/<owner>/<repo>`; `scripts/build` sets both, whatever machine builds.
- CI runs on GitHub's hosted runners only and never builds, pulls, or runs an image: it reads the registry's API, and `promote` copies within it.
- A tool directory is `tools/<name>/` with `SKILL.md`, `Containerfile`, `test/run.sh`, optionally `bin/` and more files under `test/`. `<name>` matches `^[a-z0-9]+(-[a-z0-9]+)*$`; the skill's `name` equals it; `metadata.version` matches `^\d+\.\d+\.\d+$`.
- A tool's record is `results/<name>/<version>.txt`: a `---` frontmatter block (`tool`, `version`, `image`, `digest`, `runner`, `host`, `started`, `exit`), then the test's full output. It lives outside `tools/<name>/`.
- Every Containerfile ends with `COPY test /toolshed/test` after its weights layer, sets the environment first and fetches weights last, and sets `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1` where transformers is used) so nothing downloads at run time. Pinned versions for every package and base image.
- A tool's test is `bash /toolshed/test/run.sh` inside the image: exit 0 iff the tool works; it writes only under `/tmp`. It passes with `--network none` under podman.
- `scripts/check.py` runs on a cluster's login node as well as in CI: Python 3.8 or later, stdlib only, `from __future__ import annotations`. Other scripts are bash with `set -euo pipefail`, safe under bash 3.2 (empty arrays expanded as `${a[@]+"${a[@]}"}`). The unit tests run with `python3 -I -m unittest discover tests`.
- A new tool's package is made public by hand, once, by the member whose credential first pushed it, before its candidate is tested.
- Skills are in the Agent Skills format: frontmatter `name`, `description` (in double quotes when it holds a colon), `metadata: {version}`; every example is a full command line with the program on PATH, never an entrypoint argument.
- Commit messages: one sentence stating the behavior; end with the session's `Co-Authored-By` trailer. Tasks 5 to 8 end in a merged pull request on the shed, merged with a merge commit.

## Review Focus

1. **The record of an older candidate** (the contributor rebuilt after testing, or tested before a last commit): `check` refuses with the record's digest and the candidate's. Test in Task 2 (`test_a_record_of_another_digest_is_refused`).
2. **A build from a working tree with uncommitted changes under `tools/<name>/`:** `scripts/build` refuses, naming the directory, since the tag would name source the image was not built from. Test in Task 3 (`test_build_refuses_uncommitted_changes`).
3. **An image built on an Apple Silicon Mac:** `scripts/build` passes `--platform linux/amd64`, and `check` refuses a candidate whose architecture is not `amd64`. Tests in Task 2 (`test_a_candidate_not_amd64_is_refused`) and Task 3 (`test_build_targets_amd64_with_the_source_label`).
4. **A new tool's package still private after its first push:** `scripts/push` exits 3 and prints the package's settings link; `check` reports the candidate as missing or not public. Tests in Task 2 (`test_a_candidate_not_readable_anonymously_is_reported`) and Task 3 (`test_push_says_when_the_package_is_private`); live in Task 5.
5. **`main` changed the same tool after the candidate was built**, so the merge commit's tree is one nobody built: `check` on the updated pull request finds no candidate, and `promote` fails rather than publish an untested image. Test in Task 4 (`test_a_missing_candidate_is_refused`, carried over, and the workflow computes the tag from the merge commit).

---

### Task 1: The repository, the check script, and its test

Status: done and pushed (`e185d8e`, `2c433cb`, `061da4e`). As executed, the test command is `python3 -I -m unittest discover tests` (under `-I` the file form cannot import `tests.test_check`), the test file held five tests, and `published_tags` reads GHCR's 403 for a missing package and follows its paged tag lists. Task 3 below replaces this task's `scripts/build` and `scripts/test`.

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

### Task 2: The check script reads a candidate and its record

**Files:**
- Modify: `scripts/check.py`, `tests/test_check.py`

**Interfaces:**
- Consumes: `frontmatter`, `check_tool`, `published_tags`, `NAME` from Task 1.
- Produces, in `scripts/check.py`:
  - `registry_of(remote: str) -> str`: `https://github.com/O/R(.git)` or `git@github.com:O/R(.git)` to `ghcr.io/o/r`.
  - `candidate_tag(tool_dir: Path) -> str`: `"src-" + git -C <repo root> rev-parse HEAD:tools/<name>`, with the repo root `tool_dir.parents[1]`.
  - `inspect(registry: str, name: str, tag: str) -> tuple[str, str] | None`: the manifest's digest (the `Docker-Content-Digest` header) and its architecture (from the config blob's `architecture`, or for an index the platforms' architectures joined by `,`), read anonymously as `published_tags` does; `None` when the token or the manifest is refused (401, 403, 404).
  - `check_candidate(tool_dir: Path, tag: str, candidate: tuple[str, str] | None) -> list[str]`: the reasons, from the strings below.
  - The command line: `check.py <tool dir>... [--registry R]` as before, and with `--registry` it also runs `check_candidate` with `inspect`'s answer for each tool `check_tool` found no reason against. `check.py --candidate <tool dir> [--registry R]` prints `<registry>/<name>:<candidate_tag>`. `check.py --resolve <tool dir> [--registry R]` prints `<registry>/<name>@<digest>`, or prints `candidate <tag> is not in the registry or not public` to stderr and exits 1. Without `--registry`, both take `registry_of(git remote get-url origin)`.
- The reasons, exactly (`{name}`, `{tag}`, `{version}`, `{arch}`, `{got}`, `{want}`, `{code}` filled in):
  - `candidate {tag} is not in the registry or not public; push it with scripts/push {name} and make its package public` (and no record reasons follow)
  - `candidate {tag} is {arch}, not amd64; rebuild it with scripts/build {name}`
  - `results/{name}/{version}.txt is missing; test the candidate with scripts/test {name}`
  - `results/{name}/{version}.txt is of {got}, not the candidate {want}; test the current candidate`
  - `results/{name}/{version}.txt shows exit {code}, not 0`

- [ ] **Step 1: Write the failing tests**

In `tests/test_check.py`, a helper `record(root, name="esm2-650m", version="1.0.0", digest="sha256:a", exit="0")` writing `results/<name>/<version>.txt` as `---\ntool: ...\nversion: ...\ndigest: {digest}\nexit: {exit}\n---\nok\n`. Then:

```python
class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()); self.d = tool(self.root)
    def reasons(self, candidate):
        return check.check_candidate(self.d, "src-x", candidate)
    def test_registry_of_https_and_ssh_remotes(self):
        for url in ("https://github.com/BrineyLab/ToolShed.git", "https://github.com/BrineyLab/ToolShed", "git@github.com:BrineyLab/ToolShed.git"):
            self.assertEqual(check.registry_of(url), "ghcr.io/brineylab/toolshed")
    def test_candidate_tag_is_the_committed_tree(self):
        git(self.root, "init", "-q"); git(self.root, "add", "-A"); git(self.root, "commit", "-qm", "x")
        tree = git(self.root, "rev-parse", "HEAD:tools/esm2-650m").strip()
        self.assertEqual(check.candidate_tag(self.d), f"src-{tree}")
    def test_a_good_candidate_and_record_pass(self):
        record(self.root)
        self.assertEqual(self.reasons(("sha256:a", "amd64")), [])
    def test_a_candidate_not_readable_anonymously_is_reported(self):
        self.assertEqual(self.reasons(None), ["candidate src-x is not in the registry or not public; push it with scripts/push esm2-650m and make its package public"])
    def test_a_candidate_not_amd64_is_refused(self):
        record(self.root)
        self.assertEqual(self.reasons(("sha256:a", "arm64")), ["candidate src-x is arm64, not amd64; rebuild it with scripts/build esm2-650m"])
    def test_a_missing_record_is_refused(self):
        self.assertEqual(self.reasons(("sha256:a", "amd64")), ["results/esm2-650m/1.0.0.txt is missing; test the candidate with scripts/test esm2-650m"])
    def test_a_record_of_another_digest_is_refused(self):
        record(self.root, digest="sha256:old")
        self.assertEqual(self.reasons(("sha256:a", "amd64")), ["results/esm2-650m/1.0.0.txt is of sha256:old, not the candidate sha256:a; test the current candidate"])
    def test_a_failed_record_is_refused(self):
        record(self.root, exit="1")
        self.assertEqual(self.reasons(("sha256:a", "amd64")), ["results/esm2-650m/1.0.0.txt shows exit 1, not 0"])
```

In `PublishedTagsTests` (it has the `stub` helper): `test_inspect_reads_digest_and_architecture` (the stub answers the token, the manifest with header `Docker-Content-Digest: sha256:m` and body `{"mediaType": "application/vnd.oci.image.manifest.v1+json", "config": {"digest": "sha256:c"}}`, and the blob `sha256:c` with `{"architecture": "amd64"}`; `inspect("ghcr.io/o/r", "t", "src-x") == ("sha256:m", "amd64")`), and `test_inspect_of_a_private_or_missing_candidate_is_none` (a 403 from the token, then a 404 from the manifest: `None` both times). The `git` helper is `test_changed_tools.py`'s; move it to a `tests/helpers.py` both import, with `-c commit.gpgsign=false` added.

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -I -m unittest discover tests`
Expected: FAIL: `module 'check' has no attribute 'check_candidate'` (and the other new names).

- [ ] **Step 3: Implement the five interfaces in `scripts/check.py`**

Add `from __future__ import annotations` at the top. `inspect` sends `Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json`. `check_candidate` reads the version with `frontmatter` and the record with `frontmatter` too.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -I -m unittest discover tests`
Expected: PASS, every test.

- [ ] **Step 5: Check `inspect` against live GHCR**

Run: `python3 -I -c "import sys; sys.path.insert(0, 'scripts'); import check; print(check.inspect('ghcr.io/astral-sh', 'uv', '0.1.37'), check.inspect('ghcr.io/astral-sh', 'uv', 'no-such-tag'))"`
Expected: a `sha256:` digest with an architecture (or several joined by `,`, since that tag is multi-platform), then `None`.

- [ ] **Step 6: Commit**

```bash
git add scripts/check.py tests/
git commit -m "check reads a tool's candidate from the registry and its committed record, and refuses a candidate that is missing, private, or not amd64, or a record of another image or a failed test"
```

---

### Task 3: The contributor's scripts build, push, and test a candidate

**Files:**
- Modify: `scripts/build`, `scripts/test` (both replace Task 1's)
- Create: `scripts/push`, `tests/test_scripts.py`

**Interfaces:**
- Consumes: `check.py --candidate`, `check.py --resolve`, `frontmatter` (Task 2).
- Produces:
  - `scripts/build <name>`: exits 1 with `tools/<name> has uncommitted changes; commit them first` when `git status --porcelain -- tools/<name>` prints anything. Otherwise `podman build --platform linux/amd64 --label org.opencontainers.image.source=https://github.com/<owner>/<repo> [--secret id=hf_token,env=HF_TOKEN] -t <candidate> -f tools/<name>/Containerfile tools/<name>`, with `<candidate>` from `check.py --candidate` and `<owner>/<repo>` taken from it, the secret only when `HF_TOKEN` is set.
  - `scripts/push <name>`: `podman push <candidate>`, then `check.py --resolve`. Exit 0 printing the resolved reference when it resolves. Otherwise exit 3 printing `pushed, but the package is private: make it public at https://github.com/orgs/<owner>/packages/container/<repo>%2F<name>/settings`.
  - `scripts/test <name> podman` or `scripts/test <name> srun [srun flags...]`: resolves the candidate (exit 1 if it does not), runs `bash -c 'echo "node: $(hostname)"; nvidia-smi -L 2>/dev/null || echo "gpu: none"; bash /toolshed/test/run.sh'` in it by digest, the output shown as it runs and kept. The podman form is `podman run --rm --network none [--device nvidia.com/gpu=all when nvidia-smi is on PATH] <ref>`. The srun form is `srun <flags> --container-image=<ref with its first / as #>`. Then it writes `results/<name>/<version>.txt`: frontmatter `tool`, `version`, `image` (the resolved reference), `digest`, `runner` (`podman`, or `srun` and its flags), `host` (`hostname -f`), `started` (UTC, `date -u +%Y-%m-%dT%H:%M:%SZ`), `exit`, then the output. It exits with the test's exit code.

- [ ] **Step 1: Write the failing tests**

`tests/test_scripts.py` builds a temporary repository: the real `scripts/build`, `scripts/push`, and `scripts/test`; a fake `scripts/check.py` that re-exports the real `frontmatter` (`scripts/test` imports it for the version) and, for `--candidate`, prints `ghcr.io/owner/repo/t:src-abc`, and for `--resolve` prints `ghcr.io/owner/repo/t@sha256:d` when `$FAKE_PUBLIC` is set and exits 1 otherwise; `tools/t/` with a SKILL.md at version 1.0.0, committed. Fake `podman` and `srun` on `PATH` append their arguments to `$FAKE_LOG`, print `ok`, and exit `${FAKE_EXIT:-0}`.

```python
def test_build_refuses_uncommitted_changes(self):
    (self.repo / "tools/t/SKILL.md").write_text("changed")
    r = self.run_script("build", "t")
    self.assertEqual(r.returncode, 1); self.assertIn("tools/t has uncommitted changes", r.stderr); self.assertEqual(self.log(), [])
def test_build_targets_amd64_with_the_source_label(self):
    self.assertEqual(self.run_script("build", "t").returncode, 0)
    self.assertEqual(self.log(), ["build --platform linux/amd64 --label org.opencontainers.image.source=https://github.com/owner/repo -t ghcr.io/owner/repo/t:src-abc -f tools/t/Containerfile tools/t"])
def test_push_says_when_the_package_is_private(self):
    r = self.run_script("push", "t")
    self.assertEqual(r.returncode, 3)
    self.assertIn("https://github.com/orgs/owner/packages/container/repo%2Ft/settings", r.stdout)
    self.assertEqual(self.log(), ["push ghcr.io/owner/repo/t:src-abc"])
def test_push_of_a_public_candidate_succeeds(self):
    self.assertEqual(self.run_script("push", "t", FAKE_PUBLIC="1").returncode, 0)
def test_test_runs_by_digest_and_writes_the_record(self):
    r = self.run_script("test", "t", "srun", "--partition=gpu", "--gres=gpu:1", FAKE_PUBLIC="1")
    self.assertEqual(r.returncode, 0)
    self.assertIn("--partition=gpu --gres=gpu:1 --container-image=ghcr.io#owner/repo/t@sha256:d bash -c", self.log()[0])
    rec = check.frontmatter(self.repo / "results/t/1.0.0.txt")
    self.assertEqual({k: rec[k] for k in ("tool", "version", "image", "digest", "runner", "exit")},
        {"tool": "t", "version": "1.0.0", "image": "ghcr.io/owner/repo/t@sha256:d", "digest": "sha256:d", "runner": "srun --partition=gpu --gres=gpu:1", "exit": "0"})
    self.assertIn("ok", (self.repo / "results/t/1.0.0.txt").read_text())
def test_test_records_a_failure_and_exits_with_it(self):
    r = self.run_script("test", "t", "podman", FAKE_PUBLIC="1", FAKE_EXIT="7")
    self.assertEqual(r.returncode, 7)
    self.assertEqual(check.frontmatter(self.repo / "results/t/1.0.0.txt")["exit"], "7")
def test_test_refuses_a_candidate_it_cannot_resolve(self):
    self.assertEqual(self.run_script("test", "t", "podman").returncode, 1); self.assertEqual(self.log(), [])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -I -m unittest discover tests`
Expected: FAIL: build takes two arguments, `scripts/push` does not exist, and test takes an image.

- [ ] **Step 3: Write the three scripts**

As the Interfaces block says; `#!/usr/bin/env bash`, `set -euo pipefail`, executable, `cd "$(dirname "$0")/.."` first. `scripts/test` keeps the test's exit code with `set +e` around the run and `${PIPESTATUS[0]}` after `| tee`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -I -m unittest discover tests`
Expected: PASS, every test.

- [ ] **Step 5: Run them under bash 3.2**

Run: `podman run --rm -v "$PWD":/repo:ro -w /repo docker.io/library/bash:3.2 bash -c 'apk add -q git python3 >/dev/null && python3 -I -m unittest discover tests'`
Expected: PASS (the scripts' empty arrays expand under macOS's bash).

- [ ] **Step 6: Commit**

```bash
git add scripts/ tests/
git commit -m "A contributor builds a tool's candidate for amd64 from committed source, pushes it and learns whether its package is public, and tests it by digest into a committed record"
```

---

### Task 4: The workflow checks and promotes, and the guidance says how to contribute

**Files:**
- Modify: `.github/workflows/tools.yml`, `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `.github/pull_request_template.md`
- Delete: `.github/actionlint.yaml`

**Interfaces:**
- Consumes: `check.py --registry` and `--candidate` (Task 2), `scripts/changed-tools` and `scripts/promote <repository> <candidate tag> <version tag>` (the original Task 2, unchanged).
- Produces: the jobs `check` and `promote`, nothing else; the required status check `check` on `main`.

- [ ] **Step 1: Rewrite the workflow**

Delete the `build` job and the comment above it. Permissions per job: top level `contents: read`; `promote` adds `packages: write`. `check` is as it is, its `check.py ... --registry` now covering the candidate and the record. `promote` drops the pull request lookup and, for each tool from `scripts/changed-tools "$BEFORE" "$GITHUB_SHA"` (assigned on its own line, then the empty case exits 0 with `no tool to promote`), reads `VERSION` as now, sets `TAG=$(python3 -I scripts/check.py --candidate "tools/$t" --registry "$REGISTRY/${GITHUB_REPOSITORY,,}")`, and runs `scripts/promote "${TAG%:*}" "${TAG##*:}" "$VERSION"`, adding `published: <repository>:<version> from <candidate tag>` to the step summary. The header comment says: hosted runners only; CI never builds, pulls, or runs an image.

- [ ] **Step 2: Lint the workflow**

Run: `podman run --rm -v "$PWD":/repo:ro -w /repo docker.io/rhysd/actionlint:latest -color`
Expected: no findings, with `.github/actionlint.yaml` deleted (no self-hosted label remains).

- [ ] **Step 3: Rewrite the guidance**

`CONTRIBUTING.md`: *Building and testing locally* becomes *Building, pushing, and testing*: commit, `scripts/build <name>`, a quick `podman run --rm --network none <candidate> bash /toolshed/test/run.sh`, `scripts/push <name>` (a new tool's first push prints its package's settings link: make it public there once, then `scripts/push` again), `scripts/test <name> srun --partition=<p> --gres=gpu:1 --time=00:30:00` on the cluster's login node from a clone of the branch (or `scripts/test <name> podman` where a laptop is where the tool will be used), `git add results/<name>/<version>.txt`. *The pull request* loses the candidate-built-by-CI step and the pasted output; `check` checks the candidate and the record; a member other than the builder reviews the record's output and merges; `promote` publishes. A prose-only change that rebuilds to the same digest copies its earlier record to the new version's path. *Weights behind a login*: `HF_TOKEN` in the builder's environment, never a GitHub secret. Remove the fork-approval paragraph.

`README.md`: the CI sentence under *Versions* becomes `CI checks every pull request: the tool's rules, an unpublished version, a public amd64 candidate, and a record of that candidate's passing test. On merge it copies the candidate to <name>:<version>. CI never builds or runs an image.`

`AGENTS.md`: the layout gains `scripts/push`, `results/<name>/<version>.txt`, and `scripts/test <name>`'s new meaning; under *Rules the shed keeps*, the CI line becomes `CI runs on hosted runners and never builds, pulls, or runs an image. Whoever builds a tool tests its candidate where it will be used and commits the record; an agent cannot claim a tool works without one.` and the builds line becomes `No cluster ever builds an image. The contributor builds with scripts/build, from committed source.`

`.github/pull_request_template.md`: `## Test output` becomes `## Record` (`results/<name>/<version>.txt`, where it ran); the checklist gains `Candidate pushed and its package public`.

- [ ] **Step 4: Run the tests and lint again**

Run: `python3 -I -m unittest discover tests && podman run --rm -v "$PWD":/repo:ro -w /repo docker.io/rhysd/actionlint:latest`
Expected: PASS; no findings.

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "CI checks a pull request's candidate and record and promotes on merge, on hosted runners only, and the guidance tells a contributor how to build, push, and test"
git push origin main
```

- [ ] **Step 6 [human, or an agent with the user's go-ahead]: Require `check` on `main`**

Run: `gh api -X PUT repos/brineylab/toolshed/branches/main/protection --input -` with `{"required_status_checks": {"strict": false, "contexts": ["check"]}, "enforce_admins": false, "required_pull_request_reviews": null, "restrictions": null}`.
Expected: the response names `check` under `required_status_checks.contexts`.

---

### Task 5: `hello`, and the shed's first candidate end to end

**Files:**
- Create: `tools/hello/SKILL.md`, `tools/hello/Containerfile`, `tools/hello/bin/hello`, `tools/hello/test/run.sh`, `results/hello/1.0.0.txt`

**Interfaces:**
- Consumes: Tasks 2 to 4.
- Produces: the published image `ghcr.io/brineylab/toolshed/hello:1.0.0`; the answers to two live questions, recorded in the pull request: whether slinky's enroot pulls by digest, and whether the source label gave the package's Actions access to the repository.

- [ ] **Step 1: Write the `hello` tool**

`tools/hello/SKILL.md`: `name: hello`, `description: "Checks that this machine or cluster runs toolshed images: prints the OS, the user, and the GPU it sees. Reach for it first on a new cluster, before any real tool."`, `metadata: {version: 1.0.0}`; body: Image, Commands (`hello` prints three lines: `os=<id>`, `user=<uid>`, `gpu=<nvidia-smi name or none>`), Running it (the laptop line and the `srun --container-image=ghcr.io#brineylab/toolshed/hello:1.0.0 hello` line), Gotchas (under pyxis the user is the node user, not root).

`tools/hello/Containerfile`: `FROM docker.io/library/alpine:3.20`; `RUN apk add --no-cache bash`; `COPY bin/hello /usr/local/bin/hello`; `RUN chmod +x /usr/local/bin/hello`; `COPY test /toolshed/test`. `tools/hello/bin/hello`: a `sh` script printing the three lines (`gpu=` from `nvidia-smi --query-gpu=name --format=csv,noheader` when present, else `none`). `tools/hello/test/run.sh`: `set -eu; out=$(hello); echo "$out"; case "$out" in *"os="*"user="*"gpu="*) ;; *) exit 1;; esac`.

- [ ] **Step 2: Commit on a branch, build, and try it**

```bash
git checkout -b hello && git add tools/hello
git commit -m "The hello tool checks that a machine or cluster runs toolshed images"
scripts/build hello
podman run --rm --network none "$(python3 -I scripts/check.py --candidate tools/hello)" bash /toolshed/test/run.sh
```
Expected: three lines and exit 0.

- [ ] **Step 3: Push the first candidate (Review Focus 4, live)**

Run: `scripts/push hello`
Expected: exit 3 and the line naming `https://github.com/orgs/brineylab/packages/container/toolshed%2Fhello/settings`.

- [ ] **Step 4 [human]: Make the package public and look at its Actions access**

On that page: *Change visibility* to public. Under *Manage Actions access*, note whether `brineylab/toolshed` is listed with the Write role (the source label's link); if not, add it with Write. Then `scripts/push hello` again. Expected: exit 0, printing `ghcr.io/brineylab/toolshed/hello@sha256:...`.

- [ ] **Step 5 [human, on slinky's login node]: Test the candidate**

From a clone of the branch: `scripts/test hello srun --partition=<a GPU partition from flyloft's clusters/slinky.toml> --gres=gpu:1 --time=00:10:00`
Expected: `os=alpine`, `user=<your uid>`, a GPU name, exit 0, and `results/hello/1.0.0.txt` written. If enroot refuses the `@sha256:` reference, that is the design's fallback (§2, *The record*): `scripts/test` runs the tag and records the digest it resolved before and after the job, failing when they differ. Write that change test-first in `tests/test_scripts.py`, rerun, and note it in the pull request.

- [ ] **Step 6: Commit the record and open the pull request**

```bash
git add results/hello/1.0.0.txt
git commit -m "hello's candidate passes its test on slinky"
git push -u origin hello
gh pr create --fill
```
Expected: `check` passes. The pull request's body says what Steps 4 and 5 found.

- [ ] **Step 7: Merge and confirm promotion**

`gh pr merge --merge --delete-branch`. Expected: `promote` succeeds on the merge commit; `skopeo inspect --format '{{.Digest}}' docker://ghcr.io/brineylab/toolshed/hello:1.0.0` prints the record's digest; an anonymous `podman pull ghcr.io/brineylab/toolshed/hello:1.0.0` works from a laptop with no login. If `promote` is refused writing the package, give the repository Write under *Manage Actions access* and re-run it; re-running is safe (`scripts/promote`).

---

### Task 6: `esm2-650m`

**Files:**
- Create: `tools/esm2-650m/SKILL.md`, `tools/esm2-650m/Containerfile`, `tools/esm2-650m/bin/esm2`, `tools/esm2-650m/test/run.sh`, `tools/esm2-650m/test/seqs.fasta`, `tools/esm2-650m/test/masked.fasta`, `tools/esm2-650m/test/check.py`, `results/esm2-650m/1.0.0.txt`

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

- [ ] **Step 4: Write the skill**

`SKILL.md` per the design's §1 outline: frontmatter (`name: esm2-650m`, a description that says what it is for and when to reach for it, quoted, `metadata: {version: 1.0.0}`); Image; Commands (the three, each as a full line with its inputs and outputs); Inputs and outputs (FASTA in, TSV / npz / FASTA out, what `pll` means and its sign); Compute (one GPU, memory, and a measured rate from Step 6: sequences per second for `pll` at a stated length); Running it (the laptop line from the README with a real command; the flyloft line `request_run(tool="esm2-650m@1.0.0", command="esm2 pll --in data/seqs.fasta --out $FLYLOFT_OUT/pll.tsv", gpus=1, time_limit="00:30:00")`); Gotchas (`<mask>` is the mask token; sequences longer than 1022 residues are refused by the model; CPU is fine for a few sequences).

- [ ] **Step 5: Commit on a branch, build, and try it**

Branch `esm2-650m`; commit `ESM-2 650M as a tool: pseudo-log-likelihoods, embeddings, and infilling in one image`; `scripts/build esm2-650m`; `podman run --rm --network none <candidate> bash /toolshed/test/run.sh`. Expected: exit 0 on a laptop without a GPU (CPU, under a minute) and with one; no download attempted.

- [ ] **Step 6: Push, make public, test on slinky, open the pull request**

`scripts/push esm2-650m` (exit 3: make the package public [human], push again: exit 0); on slinky, `scripts/test esm2-650m srun --partition=<p> --gres=gpu:1 --time=00:30:00` (expected exit 0, the GPU named in the record, `pll` in seconds); commit the record; push the branch; `gh pr create --fill`. Expected: `check` passes. Take Step 4's measured rate from the record and amend the skill before the pull request if it was a placeholder; a skill change changes the tree, so build, push, and test again.

- [ ] **Step 7: Merge**

A member other than the builder reads the record and merges with `--merge --delete-branch`. Expected: `promote` publishes `esm2-650m:1.0.0` with the record's digest.

- [ ] **Step 8: Verify the version rule live**

Branch `esm2-prose`: edit one sentence in `tools/esm2-650m/SKILL.md`, no version change; push the branch; open the pull request. Expected: `check` fails with `tools/esm2-650m: version 1.0.0 is already published; bump metadata.version`. Close it.

- [ ] **Step 9: Run it from a laptop**

With a `seqs.fasta` in the working directory, the skill's laptop line. Expected: `pll.tsv` written.

---

### Task 7: `esmc-600m`

**Files:**
- Create: `tools/esmc-600m/` with the same parts as Task 6 (`bin/esmc`), `results/esmc-600m/1.0.0.txt`

**Interfaces:**
- Produces: `esmc pll`, `esmc embed` (shape (n, 1152)), `esmc fill`, with the same arguments as `esm2`.

- [ ] **Step 1: Copy the sibling and change what differs**

`cp -r tools/esm2-650m tools/esmc-600m`; rename `bin/esm2` to `bin/esmc`; `test/check.py` expects 1152; `run.sh` calls `esmc`; `masked.fasta` uses `_` for a masked position.

- [ ] **Step 2: Write the Containerfile**

Same base; `pip install --no-cache-dir esm==<current>` (EvolutionaryScale's package) and its pins; weights: `RUN --mount=type=secret,id=hf_token HF_TOKEN=$(cat /run/secrets/hf_token) HF_HUB_OFFLINE=0 python -c "from esm.models.esmc import ESMC; ESMC.from_pretrained('esmc_600m')"` if the weights are gated (the first build tells: a 401 from Hugging Face), else without the secret; `ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1`; `COPY test /toolshed/test` last.

- [ ] **Step 3: Write `bin/esmc`**

The `esm` package's API: `ESMC.from_pretrained("esmc_600m").to(device)`, `ESMProtein(sequence=...)`, `model.encode`, `model.logits(..., LogitsConfig(sequence=True, return_embeddings=True))`. `pll` masks with the package's mask character (`_`) one position at a time; `embed` mean-pools the returned embeddings over residues; `fill` replaces `_` (the skill says to write `_` for a masked position in this tool).

- [ ] **Step 4: Write the skill**

As Task 6 Step 4, with the ESM-C facts: 1152-dimensional embeddings, `_` as the mask, the license is non-commercial (one line under Gotchas), and a measured rate.

- [ ] **Step 5: Build, push, test, pull request, merge**

As Task 6 Steps 5 to 7, branch `esmc-600m`, commit `ESM-C 600M as a tool: pseudo-log-likelihoods, embeddings, and infilling in one image`. If the build fails with a Hugging Face 401, export `HF_TOKEN` (a read token of a lab account that has accepted the ESM-C license) and add the `--mount=type=secret` line; record which it was in the pull request.

---

### Task 8: The scratch shed for flyloft's tier two

**Files:** none in this repository.

- [ ] **Step 1 [human]: Create `brineylab/toolshed-dev`**

A public repository created from the shed at `main` after Task 7 (`gh repo create brineylab/toolshed-dev --public`, then push the shed's `main` into it), with `check` required on its `main` as in Task 4 Step 6. Its registry is `ghcr.io/brineylab/toolshed-dev/`, its packages new, so each is made public by hand once.

- [ ] **Step 2: Confirm the flow runs there**

In a clone of `toolshed-dev`: bump `tools/hello` to `1.0.1` on a branch, commit, `scripts/build hello`, `scripts/push hello` (make public [human], push again), `scripts/test hello srun ...` on slinky, commit the record, open the pull request. Expected: `check` passes; merge; `promote` publishes `ghcr.io/brineylab/toolshed-dev/hello:1.0.1`. `toolshed-dev` is plan B's tier-two shed (flyloft's `docs/design/toolshed.md`); it is not a place for real tools.
