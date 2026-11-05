#!/bin/bash
# LIBERO eval of pi05_libero (docs/libero_plan.md Stage 0): openpi policy server + LIBERO client on one node.
# Usage: SUITE=libero_spatial TRIALS=1 NAME=smoke sbatch scripts/libero_eval.sbatch               (openpi main.py)
#        SUITE=libero_10 TASKS=8 REPEATS=2 SEED=8 NAME=moka sbatch scripts/libero_eval.sbatch  (scripts/libero_rollouts.py)
#        CLIENT_ARGS="--args.fresh-layouts 100 --args.save-raw" ...   (extra libero_rollouts.py arguments)
#        FT_RUN=<runs/policy_*> FT_TAG=positive|negative|none ...  Stage 6: serve a fine-tuned run
#        (scripts/serve_pi05_finetuned.py; FT_TAG default auto = positive for arm C, none otherwise)
#        FT_STEP=<n|final> serve trainable_<n>.msgpack (runs trained with --save-every)
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
SUITE=${SUITE:-libero_spatial} TRIALS=${TRIALS:-1} NAME=${NAME:-smoke} SEED=${SEED:-7}
OPENPI=/iris/u/ankile/karenvo/pi05/openpi
ROOT=/iris/u/ankile/karenvo/libero
CKPT=/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero
OUT=$ROOT/runs/${NAME}_${SUITE}_${SLURM_JOB_ID}
PORT=$((8100 + SLURM_JOB_ID % 800))
mkdir -p "$OUT"
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is)"
echo "suite $SUITE tasks ${TASKS:-all} repeats ${REPEATS:-} trials/task $TRIALS seed $SEED port $PORT openpi $(git -C $OPENPI rev-parse --short HEAD) out $OUT"

cd $OPENPI
if [ -n "${FT_RUN:-}" ]; then
    echo "fine-tuned run $FT_RUN tag ${FT_TAG:-auto} step ${FT_STEP:-final}"
    .venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/serve_pi05_finetuned.py --run "$FT_RUN" \
        --tag ${FT_TAG:-auto} --step ${FT_STEP:-final} --port $PORT > "$OUT/server.log" 2>&1 &
else
.venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint \
    --policy.config pi05_libero --policy.dir $CKPT > "$OUT/server.log" 2>&1 &
fi
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do   # wait up to 10 min for the server to listen
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
    kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -50 "$OUT/server.log"; exit 1; }
    sleep 5
done
echo "server up $(date -Is)"

# Own LIBERO config: ~/.libero/config.yaml points at ASPIRE's LIBERO-PRO files, not stock LIBERO.
export LIBERO_CONFIG_PATH=$ROOT/libero_cfg MUJOCO_GL=egl PYTHONPATH=$OPENPI/third_party/libero
if [ -n "${TASKS:-}" ]; then   # our client: chosen tasks, REPEATS trials per init state, per-episode videos + CSV
    $ROOT/.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/libero_rollouts.py \
        --args.port $PORT --args.task-suite-name $SUITE --args.task-ids $TASKS --args.repeats ${REPEATS:-1} \
        --args.seed $SEED --args.out-dir "$OUT" ${CLIENT_ARGS:-} 2>&1 | tee "$OUT/client.log"
else                          # openpi's stock client: every task, TRIALS trials per task
    $ROOT/.venv/bin/python examples/libero/main.py --args.host 127.0.0.1 --args.port $PORT \
        --args.task-suite-name $SUITE --args.num-trials-per-task $TRIALS --args.seed $SEED \
        --args.video-out-path "$OUT/videos" 2>&1 | tee "$OUT/client.log"
fi
echo "end $(date -Is)"
