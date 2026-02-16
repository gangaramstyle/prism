---
description: Sync cluster state via git and analyze job status, GPU availability, and logs
argument-hint: [optional: project directory name on cluster]
allowed-tools: Bash(git pull:*), Read, Glob, Grep
---

Give the user this one-liner to run on the CBICA cluster (ready for copy-paste):

```
cd ~/$ARGUMENTS && bash scripts/cluster_snapshot.sh && git add logs/ cluster_state.txt && git commit -m "logs and snapshot" && git push
```

If no argument was provided, ask the user which project to sync.

Then ask the user to confirm when they've run it. Once confirmed, run `git pull` locally, then read `cluster_state.txt` and analyze:

1. Which of your jobs are running, pending, or completed
2. GPU availability across partitions (A100, A40, P100 on CBICA; B200, MIG on Betty)
3. Any jobs stuck in pending and why (QOS limits, priority, resources)
4. Tail the most recent logs for any actively running jobs to check for errors

Present a concise status summary with actionable next steps.
