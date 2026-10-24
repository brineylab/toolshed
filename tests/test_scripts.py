import unittest, tempfile, shutil, subprocess, os, sys
from pathlib import Path
from helpers import git

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check  # noqa: E402

# check.py's --candidate and --resolve, without git remotes or the registry; frontmatter is the real one.
FAKE_CHECK = f"""import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("real_check", {str(SCRIPTS / "check.py")!r})
real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)
frontmatter = real.frontmatter
if __name__ == "__main__":
    if "--candidate" in sys.argv:
        print("ghcr.io/owner/repo/t:src-abc")
    elif os.environ.get("FAKE_PUBLIC"):
        print("ghcr.io/owner/repo/t@sha256:d")
    elif os.environ.get("FAKE_NETWORK_DOWN"):
        raise OSError("network is unreachable")
    else:
        print("candidate src-abc is not in the registry or not public", file=sys.stderr); sys.exit(4)
"""

# podman and srun: log the arguments, print a line, exit as told.
FAKE_RUNNER = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
echo ok
exit "${FAKE_EXIT:-0}"
"""

class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.repo, self.bin = self.tmp / "repo", self.tmp / "bin"
        (self.repo / "scripts").mkdir(parents=True); self.bin.mkdir()
        for s in ("build", "push", "test", "record"):
            shutil.copy(SCRIPTS / s, self.repo / "scripts")
        (self.repo / "scripts" / "check.py").write_text(FAKE_CHECK)
        for prog in ("podman", "srun"):
            (self.bin / prog).write_text(FAKE_RUNNER); (self.bin / prog).chmod(0o755)
        t = self.repo / "tools" / "t"
        (t / "test").mkdir(parents=True)
        (t / "SKILL.md").write_text("---\nname: t\ndescription: A tool.\nmetadata:\n  version: 1.0.0\n---\n")
        (t / "Containerfile").write_text("FROM x\nCOPY test /toolshed/test\n")
        (t / "test" / "run.sh").write_text("echo ok\n")
        git(self.repo, "init", "-q"); git(self.repo, "add", "-A"); git(self.repo, "commit", "-qm", "t")

    def run_script(self, script, *args, **env):
        env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "FAKE_LOG": str(self.tmp / "log"), **env}
        return subprocess.run(["bash", str(self.repo / "scripts" / script), *args], cwd=self.repo, env=env, capture_output=True, text=True)

    def log(self):
        p = self.tmp / "log"
        return p.read_text().splitlines() if p.exists() else []

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
    def test_push_does_not_call_a_registry_error_private(self):
        r = self.run_script("push", "t", FAKE_NETWORK_DOWN="1")
        self.assertNotIn(r.returncode, (0, 3))
        self.assertNotIn("private", r.stdout)
        self.assertIn("network is unreachable", r.stderr)
    def test_build_refuses_ignored_files_in_the_tool(self):
        (self.repo / ".gitignore").write_text("__pycache__/\n")
        git(self.repo, "add", ".gitignore"); git(self.repo, "commit", "-qm", "ignore")
        (self.repo / "tools/t/test/__pycache__").mkdir()
        (self.repo / "tools/t/test/__pycache__/x.pyc").write_text("x")
        r = self.run_script("build", "t")
        self.assertEqual(r.returncode, 1); self.assertIn("tools/t has uncommitted changes", r.stderr); self.assertEqual(self.log(), [])
    def test_push_of_a_public_candidate_succeeds(self):
        self.assertEqual(self.run_script("push", "t", FAKE_PUBLIC="1").returncode, 0)
    def test_test_runs_by_digest_and_writes_the_record(self):
        r = self.run_script("test", "t", "srun", "--partition=gpu", "--gres=gpu:1", FAKE_PUBLIC="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--partition=gpu --gres=gpu:1 --container-image=ghcr.io#owner/repo/t@sha256:d bash -c", self.log()[0])
        rec = check.frontmatter(self.repo / "results/t/1.0.0.txt")
        self.assertEqual({k: rec[k] for k in ("tool", "version", "image", "digest", "runner", "exit")},
            {"tool": "t", "version": "1.0.0", "image": "ghcr.io/owner/repo/t@sha256:d", "digest": "sha256:d", "runner": "srun --partition=gpu --gres=gpu:1", "exit": "0"})
        self.assertIn("ok", (self.repo / "results/t/1.0.0.txt").read_text())
    def test_test_records_a_failure_and_exits_with_it(self):
        r = self.run_script("test", "t", "podman", FAKE_PUBLIC="1", FAKE_EXIT="7")
        self.assertEqual(r.returncode, 7)
        self.assertEqual(check.frontmatter(self.repo / "results/t/1.0.0.txt")["exit"], "7")
    def test_record_writes_the_record_from_output_on_stdin(self):
        r = subprocess.run(["bash", str(self.repo / "scripts" / "record"), "t", "ghcr.io/owner/repo/t@sha256:d", "0",
                            "sbatch --partition=gpu", "slinky", "2026-10-23T12:00:00Z"],
                           input="node: gpu-1\nok\n", capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        path = self.repo / "results/t/1.0.0.txt"
        self.assertEqual(check.frontmatter(path), {"tool": "t", "version": "1.0.0", "image": "ghcr.io/owner/repo/t@sha256:d", "digest": "sha256:d",
                                                   "runner": "sbatch --partition=gpu", "host": "slinky", "started": "2026-10-23T12:00:00Z", "exit": "0"})
        self.assertTrue(path.read_text().endswith("---\nnode: gpu-1\nok\n"))
    def test_record_refuses_an_image_not_named_by_digest(self):
        r = subprocess.run(["bash", str(self.repo / "scripts" / "record"), "t", "ghcr.io/owner/repo/t:src-abc", "0", "podman", "h", "now"],
                           input="ok\n", capture_output=True, text=True)
        self.assertEqual(r.returncode, 2); self.assertFalse((self.repo / "results/t/1.0.0.txt").exists())
    def test_record_refuses_an_exit_that_is_not_a_number(self):
        r = subprocess.run(["bash", str(self.repo / "scripts" / "record"), "t", "ghcr.io/owner/repo/t@sha256:d", "ok", "podman", "h", "now"],
                           input="ok\n", capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
    def test_test_refuses_a_candidate_it_cannot_resolve(self):
        self.assertEqual(self.run_script("test", "t", "podman").returncode, 1); self.assertEqual(self.log(), [])

if __name__ == "__main__":
    unittest.main()
