---
description: Format a multi-step cluster operation as single-line commands for phone paste
argument-hint: <what you want to do on the cluster>
---

Given the user's request: $ARGUMENTS

Generate commands that can be copy-pasted from a phone on hospital VPN.

Rules:
- Each command must be a SINGLE LINE (no newlines, no backslash continuations)
- Start with `cd ~/project_name &&`
- Chain steps with `&&`
- No interactive prompts (use -y flags where needed)
- Maximum ~500 characters per line
- If the operation requires multiple steps that can't fit in one line, number them as separate single-line commands

Format the output as a numbered list of copy-paste-ready commands. Put each command in a code block for easy copying.

If the project name is ambiguous, ask the user which project.
