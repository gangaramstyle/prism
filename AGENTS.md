# [Project Name]

Multi-domain dev environment with skills for ML research, dashboards, edge apps, and automation.

## Domains & Skills
- **ML Research**: betty-cluster, git-sync-workflow | Commands: /cluster-sync, /phone-paste, /new-slurm-job

### ML Research (HPC Clusters)
Skills: `betty-cluster`, `git-sync-workflow`
Commands: `/cluster-sync`, `/phone-paste`, `/new-slurm-job`

- **Betty SuperPOD**: DGX B200 (180GB/GPU). Submit: `sbatch --partition=dgx-b200 --gpus=1`. Pyxis/Enroot containers (`--container-image=docker://...`). Python via uv. User home storage is only (50GB).
- **Git-sync workflow**: local edit → `git push` → cluster `git pull && sbatch` → commit logs → local `git pull`. Phone-paste: single-line, prefix `cd ~/project &&`, chain `&&`.
- All cluster commands must be single-line pasteable from phone over VPN.
- All SLURM workers must be idempotent.

## Tool Preferences
- Python: uv, altair (not matplotlib)

## Rules
- Never commit .env or credentials
- Cluster commands must be single-line (phone paste on VPN)
- All SLURM workers idempotent
- Read skill files in .claude/skills/ before writing domain-specific code

## Key Files
- `.claude/skills/` — Domain skills (read before coding)
- `.claude/commands/` — Slash commands for workflows
- `.claude/hooks/` — Post-edit validation
- `.claude/mcp/` — MCP server configs

See CLAUDE.md for full domain reference with patterns and examples.
