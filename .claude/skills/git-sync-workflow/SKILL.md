---
name: git-sync-workflow
description: Workflow for projects where Claude Code edits locally but code runs on a remote cluster (CBICA, Betty). Covers the push/pull/paste cycle, log analysis, and phone-friendly command formatting. Only applies to cluster workflows — other domains have full Claude Code access.
user-invocable: true
disable-model-invocation: false
---

# Git-Sync Cluster Workflow

## When This Applies

This workflow applies **only to cluster projects** (CBICA, Betty) where Claude Code cannot SSH to the cluster. The user sometimes pastes commands from their phone while on hospital VPN.

Other domains (Cloudflare, Databricks, Chrome extensions, etc.) run with full Claude Code access and do NOT use this workflow.

## The Cycle

```
Local (Claude Code)          Remote Cluster
─────────────────           ──────────────
1. Edit code
2. git push        ──────►  3. git pull && uv sync
                            4. sbatch scripts/launch_*.sh
                            5. [jobs run]
                            6. git add logs/ && git commit && git push
7. git pull        ◄──────
8. Read logs, diagnose
9. Fix code, repeat
```

## Phone-Paste Rules

When the user is on hospital VPN and pasting from their phone:

1. **Single line only** — no multi-line scripts, no backslash continuations
2. **Prefix with `cd ~/project &&`** — always start from the project directory
3. **Chain with `&&`** — ensures each step succeeds before the next
4. **Keep under ~500 characters** — phone clipboard has practical limits
5. **No interactive prompts** — use `-y` flags where needed
6. **If too long, split into numbered steps** — each a separate single-line command

### Examples

**Good (single-line, chainable):**
```bash
cd ~/radiology_education_routing && git pull && uv sync && sbatch scripts/launch_medsiglip.sh
```

**Good (snapshot + push):**
```bash
cd ~/radiology_education_routing && bash scripts/cluster_snapshot.sh && git add logs/ cluster_state.txt && git commit -m "snapshot" && git push
```

**Bad (multi-line):**
```bash
cd ~/project
git pull
uv sync
sbatch scripts/launch_medsiglip.sh
```

## Cluster State Snapshot

The `cluster_snapshot.sh` script dumps GPU/job state to `cluster_state.txt`. After syncing via git, Claude can analyze:

1. Which jobs are running, pending, or completed
2. GPU availability across partitions
3. Any jobs stuck in pending and why
4. Recent errors in log files

### Template `cluster_snapshot.sh`

```bash
#!/bin/bash
echo "=== CLUSTER SNAPSHOT $(date) ===" > cluster_state.txt
echo "" >> cluster_state.txt

echo "=== MY JOBS ===" >> cluster_state.txt
squeue --me --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" >> cluster_state.txt
echo "" >> cluster_state.txt

echo "=== GPU AVAILABILITY ===" >> cluster_state.txt
sinfo --Node -p ai -o "%12N %.6D %10P %.11T %.4c %.17G %.8m" >> cluster_state.txt 2>/dev/null
sinfo --Node -p all -o "%12N %.6D %10P %.11T %.4c %.17G %.8m" >> cluster_state.txt 2>/dev/null
echo "" >> cluster_state.txt

echo "=== RECENT JOB HISTORY (24h) ===" >> cluster_state.txt
sacct -u $USER --starttime=$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -v-24H '+%Y-%m-%dT%H:%M:%S') --format=JobID%20,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList >> cluster_state.txt 2>/dev/null
echo "" >> cluster_state.txt

echo "=== ALL RUNNING JOBS ON AI PARTITION ===" >> cluster_state.txt
squeue -p ai -o "%.12i %.10u %.20j %.10T %.12M %.12l %.20R" >> cluster_state.txt 2>/dev/null
echo "" >> cluster_state.txt

echo "Snapshot complete: cluster_state.txt"
```

## Log Analysis Playbook

After `git pull`, scan logs for common issues:

### Quick error scan
```bash
grep -ri "error\|oom\|killed\|traceback" logs/*/slurm-*.out | tail -20
```

### Common Patterns

| Log Pattern | Meaning | Action |
|-------------|---------|--------|
| `CUDA out of memory` | Model too large for GPU | Request larger GPU or reduce batch size |
| `Killed` (no other error) | System OOM killer | Increase `--mem` or `--mem-per-gpu` |
| `TimeLimit` in sacct | Hit partition time limit | Increase `--time` or use longer partition |
| `ModuleNotFoundError` | Missing Python package | `uv sync` on cluster |
| `FileNotFoundError` | Bad path in manifest | Rebuild manifest or check data mount |
| `NCCL timeout` | Multi-node communication failure | Check NCCL env vars, try different nodes |
| Exit code 137 | SIGKILL (usually OOM) | Increase memory request |
| Exit code 143 | SIGTERM (usually timeout) | Increase time limit |

### Check failed array tasks
```bash
sacct -j <arrayid> --state=FAILED -X --format=JobID%20,State,ExitCode
```

Then resubmit only failed tasks:
```bash
sbatch --array=45,67,99 scripts/launch_foo.sh
```

## What to Commit from the Cluster

| Commit | Don't Commit |
|--------|-------------|
| `logs/` (SLURM stdout/stderr) | Full result datasets |
| `cluster_state.txt` (snapshot) | Model weights / embeddings (.pt) |
| `results/` (sampled JSON, small) | Raw DICOM files |
| Code changes from debugging | Temporary/scratch files |

## Claude's Role in the Cycle

1. **Before pushing:** Ensure code changes are correct, generate commit message
2. **After user snapshots:** Analyze `cluster_state.txt` for GPU availability, suggest what to submit
3. **After user pushes logs:** Read logs, diagnose errors, suggest fixes
4. **Generate phone-paste commands:** Format multi-step operations as single lines
5. **Track progress:** Count completed vs pending items in results
