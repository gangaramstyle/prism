#!/bin/bash
set -euo pipefail

echo "=== CLUSTER SNAPSHOT $(date) ===" > cluster_state.txt
echo "" >> cluster_state.txt

echo "=== MY JOBS ===" >> cluster_state.txt
squeue --me --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" >> cluster_state.txt || true
echo "" >> cluster_state.txt

echo "=== GPU AVAILABILITY ===" >> cluster_state.txt
sinfo --Node -p dgx-b200 -o "%12N %.6D %10P %.11T %.4c %.17G %.8m" >> cluster_state.txt 2>/dev/null || true
echo "" >> cluster_state.txt

echo "=== RECENT JOB HISTORY (24h) ===" >> cluster_state.txt
sacct -u "$USER" --starttime=$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -v-24H '+%Y-%m-%dT%H:%M:%S') --format=JobID%20,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList >> cluster_state.txt 2>/dev/null || true
echo "" >> cluster_state.txt

echo "Snapshot complete: cluster_state.txt"
