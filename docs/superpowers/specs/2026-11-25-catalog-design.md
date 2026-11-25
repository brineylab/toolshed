# CI-generated `CATALOG.md` — design

Status: draft 2026-11-25, for approval (issue #79).

## 1. Intent

Agents (flyloft turns, laptop coding agents, notebooks) currently have to list `tools/` and read every `tools/*/SKILL.md` frontmatter to discover what tools exist. With 38 tools and growing, inlining all of them bloats turn prompts. A CI-generated `CATALOG.md` at the repository root gives any agent an always-current, single-file index readable from a local clone or a single raw GitHub URL, with no CLI or MCP server to install and no merge conflicts on tool PRs.

## 2. Goals, non-goals, anti-goals

**Goals**
- Deterministically generate `CATALOG.md` at the repo root from `tools/*/SKILL.md` frontmatter and `results/<name>/<version>.txt`, using a stdlib Python script (`scripts/catalog.py`) with unit tests.
- Automatically regenerate and commit `CATALOG.md` to `main` via GitHub Actions whenever `tools/**`, `results/**`, or `scripts/catalog.py` changes on `main`, and on manual `workflow_dispatch`.
- Point `README.md` (`## Finding a tool`) and `AGENTS.md` (`## Layout`) at `CATALOG.md`.
- Stay fork-friendly: nothing in `scripts/` or `.github/` names the lab, and a fork without the deploy-key secret skips the push cleanly instead of failing CI.

**Non-goals**
- Grouping by category/tags (#41) or model family (#72) — keep a flat alphabetical table until those frontmatter fields exist.
- Editing `docs/VISION.md`, `docs/DESIGN.md`, or `.github/workflows/tools.yml`.
- Truncating or rewriting tool descriptions with heuristics or LLMs.

**Anti-goals**
- Generating or gating `CATALOG.md` inside tool PRs (causes merge conflicts across concurrent PRs).
- Giving `tools.yml` / `promote` write access to repository contents, or disturbing the squash commit's `promote` check that flyloft watches.
- Infinite CI loops or empty commits.
- Tying CI pushes to a personal GitHub PAT.

## 3. Constraints

- `scripts/catalog.py` runs under `python3 -I` on Python 3.8+ with stdlib only (`from __future__ import annotations`), reusing `scripts/check.py:frontmatter`.
- `main` currently has classic branch protection requiring the `check` status check (`strict: true`, `enforce_admins: false`) and zero repository secrets, which blocks direct pushes from `GITHUB_TOKEN` (and `GITHUB_TOKEN` cannot be added to a ruleset bypass list). Replacing classic protection on `main` with an equivalent GitHub Repository Ruleset that lists a repo-scoped write deploy key as a bypass actor lets CI commit `CATALOG.md` without weakening PR checks.
- No third-party packages in the repository; keep GitHub Actions steps free of unnecessary third-party actions.

## 4. Approaches (chosen with the human, 2026-11-25)

1. **Repo-scoped deploy key + Repository Ruleset (chosen).** Migrate `main`'s `check` requirement from classic branch protection to a Repository Ruleset and add a repo-scoped write deploy key to its bypass list (private key stored in repo secret `CATALOG_DEPLOY_KEY`). A separate workflow commits `CATALOG.md` directly to `main` after tool merges. Keeps `CATALOG.md` at the repo root on `main` with zero PR conflicts, avoids personal PATs and GitHub Apps, and skips cleanly on forks without the secret.
2. **Dedicated unprotected branch (e.g., `catalog`).** `GITHUB_TOKEN` pushes directly with zero secrets and zero protection changes, and raw URL fetches work. Not chosen: a plain `git clone` of `main` (including flyloft's clone) would not have `CATALOG.md` at the repo root.
3. **Generated and verified in PRs.** Zero secrets and lives on `main`. Not chosen: concurrent tool PRs would merge-conflict on `CATALOG.md`.

## 5. Decided shape (chosen with the human, 2026-11-25)

- **Format:** Flat alphabetical Markdown table at `CATALOG.md` with columns:
  - `Tool`: relative link `[<name>](tools/<name>/SKILL.md)` (gives both GitHub UI clicks and exact relative paths to raw-text agents).
  - `Version`: backticked `metadata.version` from `SKILL.md` frontmatter.
  - `Tested on`: hardware observed in `results/<name>/<version>.txt` (GPU model name such as `NVIDIA B200`, `NVIDIA A100-SXM4-80GB`, `NVIDIA L40S`, or `CPU`; falls back to `—` if a record is absent or unrecognized).
  - `Summary`: the full `description` from `SKILL.md` frontmatter verbatim (~12.5 KB / ~3.1k tokens across 38 tools), preserving "Reach for it…" caveats (external weights, model lists, sibling comparisons) with no sentence-splitting edge cases.
- **Workflow:** Separate `.github/workflows/catalog.yml` (leaves `tools.yml` and `promote` untouched and least-privilege). Triggers on `push` to `main` for `paths: ["tools/**", "results/**", "scripts/catalog.py"]` plus `workflow_dispatch`. Uses a concurrency group to serialize rapid merges, checks for actual changes before committing, and never loops because its commit touches only `CATALOG.md`.
- **Docs:** Update `README.md` (`## Finding a tool`) to lead with `CATALOG.md` (local clone and raw URL) and `AGENTS.md` (`## Layout`) to list `CATALOG.md`, `scripts/catalog.py`, and `.github/workflows/catalog.yml`.

## 6. Left to the builder

- Exact CLI of `scripts/catalog.py`, Markdown table escaping (`|`, newlines), and unit test coverage in `tests/test_catalog.py`.
- Workflow SSH setup (inline `GIT_SSH_COMMAND` with a temp key file rather than a third-party SSH action) and actionlint verification.
- One-time repository setup during rollout (generating the deploy key, storing `CATALOG_DEPLOY_KEY`, switching `main` from classic branch protection to the equivalent Repository Ruleset with deploy-key bypass, and verifying live via `workflow_dispatch` — done with the human's go-ahead).
