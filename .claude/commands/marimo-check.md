---
description: Validate and fix marimo notebook formatting
allowed-tools: Bash(marimo check:*), Bash(uv run marimo:*)
---

Run `uv run marimo check --fix` on the current notebook to catch and automatically resolve common formatting issues, detect cycles, and identify common pitfalls.

If errors remain after the fix, report them with suggested manual fixes.
