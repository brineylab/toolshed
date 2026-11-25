"""Generate CATALOG.md from tools/*/SKILL.md frontmatter and results/<name>/<version>.txt.

    python3 -I scripts/catalog.py [--root DIR] [--output FILE]

Prints the Markdown catalog to stdout, or writes UTF-8 to FILE when --output is given.
Runs on Python 3.8 or later with stdlib only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import check  # noqa: E402

GPU_LINE = re.compile(r"GPU \d+: ([^(\n]+)")
CPU_MARKERS = ("gpu: none", "No devices found")

HEADER = [
    "# Toolshed Catalog",
    "",
    "Generated from `tools/*/SKILL.md` and `results/`. Read a tool's `SKILL.md` for image URI, commands, inputs and outputs, compute, and gotchas.",
    "",
    "| Tool | Version | Tested on | Summary |",
    "|---|---|---|---|",
]


def _body(text: str) -> str:
    """Return text after a leading `--- ... ---` frontmatter block, or text unchanged."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1:])
    return text


def tested_on(record: Path) -> str:
    """Hardware observed in a test record: GPU model name, 'CPU', or '—'."""
    if not record.is_file():
        return "—"
    body = _body(record.read_text(encoding="utf-8-sig", errors="replace"))
    m = GPU_LINE.search(body)
    if m:
        return m.group(1).strip()
    if any(marker in body for marker in CPU_MARKERS):
        return "CPU"
    return "—"


def collect(root: Path) -> list[dict[str, str]]:
    """Collect tool metadata sorted alphabetically by tool directory name."""
    tools_dir = root / "tools"
    if not tools_dir.is_dir():
        return []
    entries = []
    for tool_dir in sorted(tools_dir.iterdir(), key=lambda p: p.name):
        skill = tool_dir / "SKILL.md"
        if not tool_dir.is_dir() or not skill.is_file():
            continue
        meta = check.frontmatter(skill)
        metadata = meta.get("metadata")
        version = metadata.get("version", "") if isinstance(metadata, dict) else ""
        desc = meta.get("description", "")
        summary = desc if isinstance(desc, str) else ""
        record = root / "results" / tool_dir.name / f"{version}.txt"
        entries.append({
            "name": tool_dir.name,
            "version": version,
            "tested_on": tested_on(record),
            "summary": summary,
            "skill": f"tools/{tool_dir.name}/SKILL.md",
        })
    return entries


def _cell(value: str) -> str:
    return " ".join(value.split()).replace("|", r"\|")


def render(entries: list[dict[str, str]]) -> str:
    """Render tool entries as a Markdown catalog table with a single trailing newline."""
    lines = list(HEADER)
    for e in entries:
        lines.append(
            f"| [{e['name']}]({e['skill']}) | `{e['version']}` | {_cell(e['tested_on'])} | {_cell(e['summary'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    text = render(collect(args.root))
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
