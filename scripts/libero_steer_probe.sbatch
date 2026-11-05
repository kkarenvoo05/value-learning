#!/bin/bash
# Best-of-N steering pre-checks (docs/CURRENT.md B2 steps 1-2): base pi05_libero server + scripts/libero_steer_probe.py.
# Usage: NAME=steer_probe sbatch --dependency=afterany:<job> scripts/libero_steer_probe.sbatch
#        CLIENT_ARGS="--args.n-cand 8 --args.repeats 2" ...  (extra libero_steer_probe.py arguments)
#        SCALE=1.5 ...  v2: candidates 1..N-1 with start noise x SCALE (serves via serve_pi05_finetuned.py --base,
#                       which honours a per-request noise_scale; same pi05_libero weights and transforms)
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:1
#SBATCH --exclude=move1,move2,move3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
NAME=${NAME:-steer_probe} SEED=${SEED:-7}
INIT=${INIT:-"9 12 13 17 25 32 40 42 44 48"}   # the 10 held-out (val) init states of the policy evals
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
if [ -n "${SCALE:-}" ]; then
    .venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/serve_pi05_finetuned.py --base --port $PORT \
        > "$OUT/server.log" 2>&1 &
    CLIENT_ARGS="--args.cand-noise-scale $SCALE ${CLIENT_ARGS:-}"
else
.venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint \
    --policy.config pi05_libero --policy.dir $CKPT > "$OUT/server.log" 2>&1 &
fi
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
    kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -50 "$OUT/server.log"; exit 1; }
    sleep 5
done
echo "server up $(date -Is)"
export LIBERO_CONFIG_PATH=$ROOT/libero_cfg MUJOCO_GL=egl PYTHONPATH=$OPENPI/third_party/libero
$ROOT/.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/libero_steer_probe.py \
    --args.port $PORT --args.task-ids 8 --args.init-states $INIT --args.seed $SEED --args.out-dir "$OUT" \
    ${CLIENT_ARGS:-} 2>&1 | tee "$OUT/client.log"
echo "end $(date -Is)"
