#!/bin/bash
# CPU-only batch host for scripts/gpu_cap.sh [2026-11-19], so the GPU cap loop outlives interactive sessions (it ran
# under setsid nohup in an interactive job and stopped with it). Not counted as a GPU job by the loop (name gpu_cap).
# Usage: sbatch scripts/gpu_cap.sbatch     (log: /iris/u/ankile/karenvo/libero/logs/gpu_cap.log; one loop at a time)
#SBATCH --job-name=gpu_cap
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
cd /move/u/karenvo/Projects/value-learning
exec bash scripts/gpu_cap.sh >> /iris/u/ankile/karenvo/libero/logs/gpu_cap.log 2>&1
