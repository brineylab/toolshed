from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import catalog  # noqa: E402

SKILL = "---\nname: {name}\ndescription: {description}\nmetadata:\n  version: {version}\n---\n# {name}\n"
RECORD = "---\ntool: {name}\nversion: {version}\nexit: 0\n---\n{body}\n"


def write_tool(root: Path, name: str, version: str = "1.0.0", description: str = "A tool.", record_body: str | None = "gpu: none") -> None:
    tool_dir = root / "tools" / name
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "SKILL.md").write_text(SKILL.format(name=name, description=description, version=version), encoding="utf-8")
    if record_body is not None:
        rec = root / "results" / name / f"{version}.txt"
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(RECORD.format(name=name, version=version, body=record_body), encoding="utf-8")


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def test_tested_on_parses_gpu_cpu_and_fallbacks(self):
        cases = [
            ("b200.txt", "node: n0\nGPU 0: NVIDIA B200 (UUID: GPU-123)\nok", "NVIDIA B200"),
            ("l40s.txt", "[2026-11-22T21:16:36Z] gpu-swann-0: GPU 0: NVIDIA L40S (UUID: GPU-456)\nok", "NVIDIA L40S"),
            ("cpu_none.txt", "node: n0\ngpu: none\nok", "CPU"),
            ("cpu_nodrv.txt", "node: n0\nNo devices found.\nok", "CPU"),
            ("unknown.txt", "node: n0\nok", "—"),
        ]
        for fname, body, expected in cases:
            p = self.root / fname
            p.write_text(RECORD.format(name="t", version="1.0.0", body=body), encoding="utf-8")
            self.assertEqual(catalog.tested_on(p), expected, fname)
        self.assertEqual(catalog.tested_on(self.root / "missing.txt"), "—")

    def test_escapes_pipes_and_collapses_whitespace(self):
        out = catalog.render([{
            "name": "t",
            "version": "1.0.0",
            "tested_on": "CPU",
            "summary": "  a | b\n  c  ",
            "skill": "tools/t/SKILL.md",
        }])
        self.assertIn(r"| [t](tools/t/SKILL.md) | `1.0.0` | CPU | a \| b c |", out)

    def test_ignores_non_tool_entries_under_tools(self):
        write_tool(self.root, "alpha")
        (self.root / "tools" / ".DS_Store").write_text("junk", encoding="utf-8")
        (self.root / "tools" / "empty").mkdir()
        entries = catalog.collect(self.root)
        self.assertEqual([e["name"] for e in entries], ["alpha"])

    def test_renders_sorted_deterministic_table_and_writes_output(self):
        write_tool(self.root, "zeta", version="2.1.0", description='"Zeta: second."', record_body="GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-1)")
        write_tool(self.root, "alpha", version="1.0.0", description='"Alpha: first."', record_body="gpu: none")
        entries = catalog.collect(self.root)
        self.assertEqual([e["name"] for e in entries], ["alpha", "zeta"])
        rendered = catalog.render(entries)
        self.assertTrue(rendered.endswith("\n"))
        self.assertFalse(rendered.endswith("\n\n"))
        lines = rendered.splitlines()
        self.assertEqual(lines[0], "# Toolshed Catalog")
        self.assertEqual(lines[4], "| Tool | Version | Tested on | Summary |")
        self.assertEqual(lines[6], "| [alpha](tools/alpha/SKILL.md) | `1.0.0` | CPU | Alpha: first. |")
        self.assertEqual(lines[7], "| [zeta](tools/zeta/SKILL.md) | `2.1.0` | NVIDIA A100-SXM4-80GB | Zeta: second. |")

        out_path = self.root / "CATALOG.md"
        res = subprocess.run(
            [sys.executable, "-I", str(REPO_ROOT / "scripts" / "catalog.py"), "--root", str(self.root), "--output", str(out_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.stdout, "")
        self.assertEqual(out_path.read_text(encoding="utf-8"), rendered)

    def test_real_repo_catalog_has_no_unknown_hardware(self):
        tool_dirs = sorted(p.name for p in (REPO_ROOT / "tools").iterdir() if p.is_dir() and (p / "SKILL.md").is_file())
        entries = catalog.collect(REPO_ROOT)
        self.assertEqual([e["name"] for e in entries], tool_dirs)
        self.assertGreater(len(entries), 0)
        for e in entries:
            self.assertTrue(e["version"], e["name"])
            self.assertTrue(e["summary"], e["name"])
            self.assertNotEqual(e["tested_on"], "—", f"{e['name']} missing hardware info")


if __name__ == "__main__":
    unittest.main()
