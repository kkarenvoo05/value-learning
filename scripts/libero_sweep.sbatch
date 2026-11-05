#!/bin/bash
# LIBERO task sweep (docs/libero_plan.md Stage 1 revisit): one policy server, several suites run one after another
# with scripts/libero_rollouts.py, init states 0..N-1 x 1 trial per task (= openpi main.py's trial i -> init state i).
# Usage: SPECS="libero_90:0-49" NAME=sweep90a sbatch scripts/libero_sweep.sbatch
#        SPECS="libero_90:50-89 libero_object:all libero_goal:all" NAME=sweep90b sbatch scripts/libero_sweep.sbatch
# SPEC = suite:all or suite:a-b (task ids, inclusive). Output: runs/<NAME>_<job>/<suite>_<a-b>/episodes.csv + videos.
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
SPECS=${SPECS:?} NAME=${NAME:-sweep} SEED=${SEED:-7} N_INIT=${N_INIT:-20}
OPENPI=/iris/u/ankile/karenvo/pi05/openpi
ROOT=/iris/u/ankile/karenvo/libero
CKPT=/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero
OUT=$ROOT/runs/${NAME}_${SLURM_JOB_ID}
PORT=$((8100 + SLURM_JOB_ID % 800))
mkdir -p "$OUT"
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is)"
echo "specs $SPECS init states 0..$((N_INIT - 1)) seed $SEED port $PORT openpi $(git -C $OPENPI rev-parse --short HEAD) out $OUT"

cd $OPENPI
.venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint \
    --policy.config pi05_libero --policy.dir $CKPT > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do   # wait up to 10 min for the server to listen
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
    kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -50 "$OUT/server.log"; exit 1; }
    sleep 5
done
echo "server up $(date -Is)"

export LIBERO_CONFIG_PATH=$ROOT/libero_cfg MUJOCO_GL=egl PYTHONPATH=$OPENPI/third_party/libero
for spec in $SPECS; do
    suite=${spec%%:*} range=${spec#*:}
    tasks=() sub=$OUT/${suite}_${range}
    [ "$range" != all ] && tasks=(--args.task-ids $(seq ${range%-*} ${range#*-}))
    echo "== $suite tasks $range $(date -Is)"
    $ROOT/.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/libero_rollouts.py \
        --args.port $PORT --args.task-suite-name $suite "${tasks[@]}" --args.init-states $(seq 0 $((N_INIT - 1))) \
        --args.repeats 1 --args.seed $SEED --args.out-dir "$sub" > "$sub.client.log" 2>&1 \
        || echo "client failed for $spec (see $sub.client.log)"
    grep -h "Task " "$sub.client.log" | sed "s/^/$suite /" || true
done
echo "end $(date -Is)"
