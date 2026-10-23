"""A tool directory's rules, and whether its version is already published.

    python3 -I scripts/check.py tools/<name>... [--registry ghcr.io/<owner>/<repo>]

Exits 0 when every directory passes, else prints one `tools/<name>: <reason>`
line per failure and exits 1. Without --registry the version rule is skipped.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def frontmatter(path: Path) -> dict:
    """The first `---` block: `key: value` lines, and `key:` opening a dict of two-space-indented lines."""
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out, nested = {}, None
    for line in lines[1:]:
        if line.strip() == "---":
            return out
        if not line.strip():
            continue
        if line.startswith("  ") and nested is not None:
            key, _, value = line.strip().partition(":")
            nested[key.strip()] = value.strip()
            continue
        key, _, value = line.partition(":")
        if value.strip():
            out[key.strip()], nested = value.strip(), None
        else:
            nested = out[key.strip()] = {}
    return {}  # no closing ---


def check_tool(tool_dir: Path, published: set[str] | None) -> list[str]:
    """The reasons this directory breaks the rules; empty when it passes."""
    name, reasons = tool_dir.name, []
    if not NAME.match(name):
        reasons.append(f"name {name!r} is not lowercase words joined by hyphens")
    skill = tool_dir / "SKILL.md"
    meta = frontmatter(skill) if skill.is_file() else None
    if meta is None:
        reasons.append("SKILL.md is missing")
    else:
        if meta.get("name") != name:
            reasons.append(f"SKILL.md name {meta.get('name')!r} is not the directory's name {name!r}")
        if not meta.get("description"):
            reasons.append("SKILL.md description is empty")
    metadata = meta.get("metadata") if meta else None
    version = metadata.get("version") if isinstance(metadata, dict) else None
    if meta is not None and not (isinstance(version, str) and VERSION.match(version)):
        reasons.append(f"SKILL.md metadata.version {version!r} is not MAJOR.MINOR.PATCH")
    containerfile = tool_dir / "Containerfile"
    if not containerfile.is_file():
        reasons.append("Containerfile is missing")
    elif "COPY test /toolshed/test" not in containerfile.read_text():
        reasons.append("Containerfile does not COPY test /toolshed/test")
    if not (tool_dir / "test" / "run.sh").is_file():
        reasons.append("test/run.sh is missing")
    if published is not None and version in published:
        reasons.append(f"version {version} is already published; bump metadata.version")
    return reasons


def published_tags(registry: str, name: str) -> set[str]:
    """GHCR's tags for <registry>/<name>, anonymously; set() when the package does not exist.

    GHCR answers a package that does not exist with 403 at the token endpoint, and pages
    tags through the Link header.
    """
    host, _, repo = registry.partition("/")
    try:
        with urllib.request.urlopen(f"https://{host}/token?scope=repository:{repo}/{name}:pull") as r:
            token = json.load(r)["token"]
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return set()
        raise
    tags, path = set(), f"/v2/{repo}/{name}/tags/list?n=1000"
    while path:
        req = urllib.request.Request(f"https://{host}{path}", headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req) as r:
            tags |= set(json.load(r).get("tags") or [])
            link = re.match(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link") or "")
        path = link and link.group(1)
    return tags


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tools", nargs="*", type=Path)
    p.add_argument("--registry")
    args = p.parse_args()
    failed = False
    for d in args.tools:
        ask = args.registry and NAME.match(d.name)  # an invalid name is already a failure, and GHCR rejects it
        published = published_tags(args.registry.lower(), d.name) if ask else None
        for reason in check_tool(d, published):
            print(f"tools/{d.name}: {reason}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
