---
name: betty-cluster
description: PARCC Betty HPC cluster reference for ML research. Covers DGX B200 GPUs, SLURM partitions, VAST/Ceph storage, containers (Pyxis/Enroot), and multi-node training. Use when discussing Betty jobs, GPU resources, or the local-edit → push → pull → run → commit-logs workflow on PARCC.
user-invocable: true
disable-model-invocation: false
---

# PARCC Betty Cluster Reference

## Workflow: Local Dev + Cluster Execution

Claude Code runs locally. The cluster runs jobs. Code and logs sync via git.

```
Local (Claude Code)          Cluster (Betty)
─────────────────           ────────────────
Edit code                   git pull
git push         ──────►    micromamba run -n env python ...
                            sbatch scripts/launch_*.sh
Read logs        ◄──────    git add logs/ results/
Diagnose & fix              git commit && git push
```

**Key principle:** Claude cannot SSH to the cluster. All debugging happens by reading committed log files and result samples locally.

### Phone-paste workflow (VPN only)
When on hospital VPN, the user pastes commands from their phone. All cluster commands must be **single-line**. Always prefix with `cd ~/project_name &&`.

### What to commit back from the cluster
- `logs/` — SLURM stdout/stderr (small, essential for debugging)
- `results/` — Sampled outputs (JSON, NOT full datasets)
- Do NOT commit large binary files (embeddings .pt, full result sets)

## Hardware

### GPU System — DGX B200 SuperPOD
- **31 DGX nodes**, each with:
  - 8× NVIDIA B200 GPUs (180 GB HBM3e each, 1.4 TB total per node)
  - 2× Intel Xeon Platinum 8570 (112 cores total)
  - 2 TB DDR5 memory
  - 30.72 TB NVMe local storage
  - NDR400 InfiniBand (8 connections per node)
- **Cluster peak:** 8.5 PFLOPs
- **Per-node peak:** 274 TFLOPs

### CPU Standard Memory
- 64 compute nodes, AMD EPYC 9374F (64 cores each)
- 384 GB DDR5 per node
- 800 GB NVMe local, NDR200 InfiniBand

### CPU Large Memory
- 10 nodes, same CPUs as standard
- 1,152 GB DDR5 per node

## SLURM Partitions

| Partition | Hardware | Limit | Use Case |
|-----------|----------|-------|----------|
| `dgx-b200` | 8× B200 per node (180GB each) | 32 GPUs | Large ML training, multi-node |
| `dgx-b200-mig90` | MIG 90GB VRAM slices | 8 MIGs | Medium GPU tasks, single-model inference |
| `dgx-b200-mig45` | MIG 45GB VRAM slices | 8 MIGs | Small GPU tasks, light inference |
| `genoa-std-mem` | AMD EPYC 64-core | 640 CPUs | CPU-only workloads, data processing |
| `genoa-large-mem` | AMD EPYC + 1.1TB RAM | 128 CPUs | Memory-heavy analysis |

### GPU Request Syntax

```bash
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1                    # Number of GPUs (not type-specific like CBICA)
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=128G
```

For MIG partitions:
```bash
#SBATCH --partition=dgx-b200-mig90
#SBATCH --gpus=1
```

**Key difference from CBICA:** Use `--gpus=N` (not `--gpus-per-node=type:N`). No need to specify GPU type — the partition determines hardware.

### Multi-Node Training

Request resources in proportional increments (8 GPUs per node):
```bash
#SBATCH --nodes=2
#SBATCH --gpus=16                   # 8 per node × 2 nodes
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=16
```

**NCCL environment variables (required for multi-node):**
```bash
export NCCL_NVLS_ENABLE=1 NCCL_IB_ADAPTIVE_ROUTING=1 NCCL_IB_SL=1 NCCL_IB_QPS_PER_CONNECTION=2 NCCL_IB_SPLIT_DATA_ON_QPS=0 NCCL_SOCKET_IFNAME=bond0 NCCL_ALGO=RING UCX_TLS=rc NCCL_IB_HCA=mlx5_15,mlx5_10,mlx5_14,mlx5_13,mlx5_8,mlx5_7,mlx5_9,mlx5_4
```

**Launch with srun (avoids shell activation issues):**
```bash
srun micromamba run -n my-env python train.py
```

PyTorch Lightning auto-detects SLURM variables (`SLURM_NTASKS`, `SLURM_NODEID`).

### Job Arrays

```bash
#SBATCH --array=0-99
# Each task gets SLURM_ARRAY_TASK_ID (0-99)
# Throttle: --array=0-99%10 (max 10 concurrent)
```

Test single task: `sbatch --array=0 scripts/launch_foo.sh`

## Storage

| Mount | Path | Quota | Notes |
|-------|------|-------|-------|
| Home | `/vast/home/<first-letter>/<PennKey>` | 50 GB, 250K inodes | Bi-weekly snapshots. **DO NOT fill** — blocks SSH login |
| Projects | `/vast/projects/<PI-PennKey>/default` | 50 GB (expandable) | High-perf VAST flash. Request more via ColdFront |
| Archive | `/ceph/projects/<PI-PennKey>/<project>` | Request-based | HDD-backed with NVMe cache. Long-term storage |
| Scratch | `/tmp/$USER` | ~500 GB shared per node | **Auto-deleted when job ends**. Copy results out first |

### Quota commands
```bash
parcc_quota.py              # Check home directory quota
parcc_du.py /path/to/dir    # Disk usage breakdown
```

### Best practices
- Code and configs → Home (`/vast/home/...`)
- Active datasets and results → Projects (`/vast/projects/...`)
- Archived data → Ceph (`/ceph/projects/...`)
- Temp I/O during jobs → Scratch (`/tmp/$USER`)
- Always back up critical files externally (GitHub, cloud)

## Login

```bash
# Requires Penn VPN (GlobalProtect)
kinit <PennKey>@UPENN.EDU
ssh <PennKey>@login.betty.parcc.upenn.edu
```

- Three login nodes: `login01`, `login02`, `login03` (or just `login.betty.parcc.upenn.edu`)
- **2-of-3 authentication:** Kerberos + SSH key + Duo (any two)
- Kerberos tickets expire after 10 hours — renew with `kinit`
- SSH multiplexing recommended for multiple sessions

**Do NOT run compute on login nodes.** Use SLURM for everything beyond quick inspection.

## Software Environment

- **OS:** DGX OS / Ubuntu 24.04 Pro
- **Scheduler:** SLURM + Run:AI
- **Modules:** Lmod (`module avail`, `module spider cuda`, `module load anaconda3`)
- **CUDA modules:** `cuda/12.8.1`, `cuda/12.9.0`, `cuda/13.1.0` (default). Load before building CUDA extensions (e.g., flash-attn).
- **Python:** `uv` preferred (install manually). Fallback: `module load anaconda3` → micromamba/conda
- **Compilers:** nvhpc, Intel, GCC
- **MPI:** Intel MPI, OpenMPI, NCCL

### Python environment setup
```bash
module load anaconda3
micromamba create -n my-env python=3.11 pytorch torchvision -c pytorch -c nvidia
micromamba activate my-env
```

**Note:** `uv` is not pre-installed on Betty. Install manually:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Important:** After `module load`, uv may disappear from PATH. Always add it back:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

Consider adding this to `~/.bashrc` to make it permanent.

## Containers (Pyxis + Enroot)

Betty uses **Pyxis/Enroot**, NOT Singularity/Apptainer.

### Run a container in a SLURM job
```bash
srun --partition=dgx-b200 --gpus=1 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.01-py3 \
     python train.py
```

### In a batch script
```bash
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:24.01-py3
#SBATCH --container-mounts=/vast/home/v/vineethg:/workspace

python train.py
```

### Mount paths
- `--container-mounts=host_path:container_path`
- `--container-mount-home` (auto-mounts home)

### NVIDIA NGC containers
Requires API key in `$HOME/.config/enroot/.credentials`:
```
machine nvcr.io login $oauthtoken password <NGC-API-KEY>
```

## Common SLURM Commands

```bash
# Submit and monitor
sbatch my_job.sh
squeue -u $USER
scontrol show job <jobid>

# Post-completion
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,AllocCPUs,ReqMem

# Cancel
scancel <jobid>
scancel -u $USER

# Interactive session (CPU)
srun --partition=genoa-std-mem --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

# Interactive session (GPU)
srun --partition=dgx-b200 --gpus=1 --cpus-per-gpu=16 --mem-per-gpu=128G --time=01:00:00 --pty bash
```

### Job states
- **PD:** Pending (awaiting resources)
- **R:** Running
- **CD:** Completed
- **F:** Failed
- **TO:** Timed out

## Key Differences from CBICA

| Aspect | CBICA | Betty |
|--------|-------|-------|
| GPUs | A100 (80GB), A40 (48GB), P100 (12GB) | B200 (180GB) |
| GPU request | `--gpus-per-node=a100:1` (type required) | `--gpus=1` (partition determines type) |
| Containers | Not documented | Pyxis + Enroot (`--container-image=docker://...`) |
| Python | `uv` pre-configured | `module load anaconda3` → micromamba |
| Storage | GPFS (`/cbica/`, `/gpfs/`) | VAST + Ceph (`/vast/`, `/ceph/`) |
| InfiniBand | HDR | NDR400 |
| Multi-node | Rarely used | Native DGX SuperPOD with NVLink/NVSwitch |
| Module system | Requires `module unload sge && module load slurm` | Clean Lmod, no SGE conflict |

## Gotchas

### 1. Home directory full = can't SSH
Keep home under 50GB. Move data to `/vast/projects/` or `/ceph/projects/`.

### 2. Scratch is ephemeral
`/tmp/$USER` is deleted when the job ends. Always copy results to persistent storage before the job finishes.

### 3. MIG partitions
MIG slices (90GB, 45GB) give you a fraction of a B200. Good for inference and small training. You cannot request multiple MIG slices to combine them.

### 4. Container image caching
First pull of a Docker image is slow (converted to SquashFS). Subsequent runs use the cache. Pre-pull images in a short interactive job if you want fast batch submissions.

### 5. Kerberos expiry
Tickets expire after 10 hours. If SSH drops, run `kinit <PennKey>@UPENN.EDU` again.

### 6. 8-GPU proportional requests
For multi-node, always request in multiples of 8 GPUs. Requesting 3 GPUs across 2 nodes will fail or waste resources.

### 7. Disable lock files for multi-node conda
When multiple nodes activate the same conda env simultaneously, lock conflicts can occur. Disable lock files in micromamba configuration for multi-node jobs.

### 8. SLURM auto-sets QOS
Betty auto-defaults `--qos=dgx` for the `dgx-b200` partition. No need to specify it manually.

### 9. Git credential caching
HTTPS clones prompt for username/PAT every time. Fix with:
```bash
git config --global credential.helper store
```
Then push/pull once with credentials — they're saved to `~/.git-credentials`.

### 10. Home path format
Home directories are at `/vast/home/<first-letter>/<PennKey>` (e.g., `/vast/home/g/gangaram`).

### 11. Building CUDA extensions on login nodes
Login nodes have no GPU but `module load cuda/12.8.1` provides `nvcc`. Required for building packages like `flash-attn`. Always load the CUDA module before `uv pip install` or `pip install` of CUDA extensions.

### 12. `module load` clobbers PATH
After `module load`, custom tools (uv, etc.) in `~/.local/bin` may vanish. Always run `export PATH="$HOME/.local/bin:$PATH"` after loading modules, or add it to `~/.bashrc`.

## Example Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=128G
#SBATCH --time=04:00:00

module load anaconda3
srun micromamba run -n my-env python src/train.py \
    --data_dir /vast/projects/pi-pennkey/default/data \
    --output_dir /vast/projects/pi-pennkey/default/results \
    --batch_size 32
```

## Cluster Utilization Intelligence

Data from 3-month historical analysis (29,775 jobs, 303,165 GPU-hours). Refresh with:
```bash
cd ~/nvreason && source .venv/bin/activate && python scripts/cluster_history.py --days 90 --output data/cluster_history.txt
```

### Best Submission Windows (data-driven)

| Rank | Time | Avg GPU-hrs/day | Notes |
|------|------|----------------|-------|
| 1 | **7 AM** | 41 | Absolute quietest hour |
| 2 | **4 AM** | 64 | Early morning lull |
| 3 | **9 AM** | 66 | Before morning rush |
| 4 | **5 AM** | 68 | |
| 5 | **1 AM** | 89 | Late night gap |

**Worst:** 12 PM (303), 2 PM (244), 1 PM (199). Lunch-hour rush is 3-5x busier than early morning.

**Best day+hour combos:** Saturday 7 AM (nearly empty), Sunday 8 AM, Saturday 6-8 AM.

**Strategy:** Submit big jobs Friday evening or Saturday morning. Weekdays aim for 4-7 AM.

### Day of Week Patterns

| Day | Avg GPU-hrs/wk | Notes |
|-----|---------------|-------|
| Saturday | 2,323 | **Quietest** |
| Sunday | 2,455 | |
| Thursday | 2,798 | |
| Wednesday | 3,357 | |
| Friday | 3,732 | Drops off after lunch |
| Monday | 3,842 | |
| Tuesday | **5,073** | **Busiest** |

### Realistic GPU Availability

| GPUs | Nodes | Feasibility | When |
|------|-------|-------------|------|
| 8 (1 node) | 1 | Almost guaranteed | Any time |
| 16 (2 nodes) | 2 | Very likely | Off-peak or before 10 AM |
| 32 (4 nodes) | 4 | Possible | Weekends, early mornings |
| 64+ (8 nodes) | 8+ | Unlikely | Only holidays |

**Typical state:** 20-22 active nodes, 4-6 idle, 1 offline (of 27 total).

### Weekly Trend

Cluster utilization doubled from Nov 2025 (~15K GPU-hrs/wk) to Jan 2026 (~30K GPU-hrs/wk). Expect continued growth — submit early and use checkpointing.

### Top Users (as of Feb 2026)

The top 3 users consume ~32% of all GPU time. Heavy users often run multi-day interactive sessions that tie up full nodes. Check `squeue -p dgx-b200` before planning multi-node jobs.

### Job Duration Patterns

- 54% of jobs are under 10 minutes (test/debug runs)
- Only 2.8% run longer than 1 day
- Cluster churns quickly — even when busy, short jobs get scheduled fast

### B200 vs H100 Performance

When comparing to papers that used H100s:
- B200 has **2.25x** more VRAM (180GB vs 80GB) — bigger batches, less gradient accumulation
- B200 is **~2.3x** faster in FP16/BF16 compute
- B200 has **2.4x** memory bandwidth
- **Rule of thumb:** 8 B200s ≈ 16-20 H100s in effective throughput

## Cluster State Snapshot (for Claude Code)

Use `scripts/cluster_snapshot.sh` to dump GPU/job state to `cluster_state.txt`, then sync via git:

**On the cluster (single-line, phone-paste ready):**
```bash
cd ~/project_name && bash scripts/cluster_snapshot.sh && git add cluster_state.txt logs/ && git commit -m "snapshot" && git push
```

Then locally, Claude runs `git pull` and reads `cluster_state.txt` to advise on GPU availability, job status, and next steps.

### Refreshing Historical Data

Run the full 90-day analysis periodically (monthly recommended):
```bash
cd ~/nvreason && source .venv/bin/activate && python scripts/cluster_history.py --days 90 --output data/cluster_history.txt && git add data/cluster_history.txt && git commit -m "refresh cluster history" && git push
```

The report covers: hourly heatmap, day-of-week patterns, weekly trend, top users, best submission windows, and job duration distribution. Claude reads this locally after `git pull` to give data-driven scheduling advice.
