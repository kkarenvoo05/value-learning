#!/bin/bash
# CPU: per-layer frozen-feature analysis (scripts/libero_frozen_encoders.py analyze) on a P1c/Step-0 hidden-state dir.
# Usage: FEAT=<dir> FIT=<fit_fixed_fresh.txt|fit_fixed.txt|fit_fresh375_s0.txt> TAG=<tag> sbatch scripts/layer_analyze_cpu.sbatch
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
cd /move/u/karenvo/Projects/value-learning
echo "host $(hostname) CPU start $(date -Is) feat $FEAT fit $FIT tag $TAG"
PYTHONPATH=scripts:src .venv/bin/python scripts/libero_frozen_encoders.py analyze --out $FEAT --fit-subset $FIT --tag $TAG \
  --encoders ${ENCODERS:-llm_h0 llm_h1 llm_h2 llm_h3 llm_h4 llm_h5 llm_h7 llm_h9 llm_h11 llm_h13 llm_h15 llm_h17 llm_h18 llm_hfinal}
echo "end $(date -Is)"
