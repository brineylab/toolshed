import unittest, tempfile, shutil, subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "changed-tools"

def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout

class ChangedToolsTests(unittest.TestCase):
    def test_changed_tools_that_still_exist_once_each(self):
        repo = Path(tempfile.mkdtemp())
        (repo / "scripts").mkdir()
        shutil.copy(SCRIPT, repo / "scripts")
        for t in ("kept", "edited", "retired"):
            (repo / "tools" / t / "test").mkdir(parents=True)
            (repo / "tools" / t / "SKILL.md").write_text(t)
        git(repo, "init", "-q", "-b", "main")
        git(repo, "add", "-A"); git(repo, "commit", "-qm", "base")
        (repo / "tools" / "edited" / "SKILL.md").write_text("changed")
        (repo / "tools" / "edited" / "test" / "run.sh").write_text("echo ok")
        shutil.rmtree(repo / "tools" / "retired")
        (repo / "tools" / "new").mkdir()
        (repo / "tools" / "new" / "SKILL.md").write_text("new")
        (repo / "tools" / "README").write_text("not a tool")
        git(repo, "add", "-A"); git(repo, "commit", "-qm", "change")
        out = subprocess.run(["bash", str(repo / "scripts" / "changed-tools"), "HEAD~1", "HEAD"],
                             check=True, capture_output=True, text=True).stdout
        self.assertEqual(out.split(), ["edited", "new"])

if __name__ == "__main__":
    unittest.main()
