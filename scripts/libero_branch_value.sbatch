#!/bin/bash
# Ground-truth value by simulator branching (scripts/libero_branch_value.py): base pi05_libero server + client.
# Usage: EPISODES="165 350 ..." [CLIENT_ARGS="--args.k 6 --args.stride 20"] sbatch scripts/libero_branch_value.sbatch
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris[1-4,9],iris-hgx-[1-2],iris-hp-z8   # Turing GPUs fail the pi05 bf16 self-check; iris9: JAX finds no CUDA device (2026-11-05); hgx: shared H100s not needed, runs on 20 GB A4500s (2026-11-08)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
: "${EPISODES:?set EPISODES (moka table episode_index list)}"
NAME=${NAME:-branch_value}
OPENPI=/iris/u/ankile/karenvo/pi05/openpi
ROOT=/iris/u/ankile/karenvo/libero
CKPT=/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero
OUT=$ROOT/runs/${NAME}_libero_10_${SLURM_JOB_ID}
PORT=$((8100 + SLURM_JOB_ID % 800))
mkdir -p "$OUT"
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) out $OUT"
cd $OPENPI
.venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint \
    --policy.config pi05_libero --policy.dir $CKPT > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
    kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -50 "$OUT/server.log"; exit 1; }
    sleep 5
done
echo "server up $(date -Is)"
export LIBERO_CONFIG_PATH=$ROOT/libero_cfg MUJOCO_GL=egl PYTHONPATH=$OPENPI/third_party/libero
$ROOT/.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/libero_branch_value.py \
    --args.port $PORT --args.episodes $EPISODES --args.out-dir "$OUT" \
    ${CLIENT_ARGS:-} 2>&1 | tee "$OUT/client.log"
echo "end $(date -Is)"
