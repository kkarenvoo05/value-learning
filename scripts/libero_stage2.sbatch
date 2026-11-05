#!/bin/bash
# Stage 2 rollouts (docs/libero_plan.md): one policy server, our client with --args.save-raw on the train then the
# val init states of configs/libero_moka.yaml, then the open-loop replay check (scripts/libero_replay.py).
# One job on purpose: the server's noise stream starts at jax.random.key(0), so two servers would replay the
# same noise sequence.
# Usage: sbatch scripts/libero_stage2.sbatch                          (full: repeats from the config)
#        REPEATS_TRAIN=1 REPEATS_VAL=0 MAX_STATES=2 NAME=s2smoke sbatch scripts/libero_stage2.sbatch   (smoke)
#        REPLAY=all|<n>  replay every episode, or the first n of each split (default 50)
#SBATCH --job-name=job
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
NAME=${NAME:-stage2} REPLAY=${REPLAY:-50}
VL=/move/u/karenvo/Projects/value-learning
OPENPI=/iris/u/ankile/karenvo/pi05/openpi
ROOT=/iris/u/ankile/karenvo/libero
CKPT=/iris/u/rbhowmik/cache/openpi/openpi-assets/checkpoints/pi05_libero
cfg() { $VL/.venv/bin/python -c "import yaml; c = yaml.safe_load(open('$VL/configs/libero_moka.yaml')); print($1)"; }
SUITE=$(cfg "c['suite']") TASK=$(cfg "c['task_id']") SEED=$(cfg "c['rollouts']['env_seed']")
OUT=$ROOT/runs/${NAME}_moka_${SLURM_JOB_ID}
PORT=$((8100 + SLURM_JOB_ID % 800))
mkdir -p "$OUT"
cp $VL/configs/libero_moka.yaml "$OUT/"
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is)"
echo "suite $SUITE task $TASK seed $SEED port $PORT openpi $(git -C $OPENPI rev-parse --short HEAD) out $OUT"

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

# Own LIBERO config: ~/.libero/config.yaml points at ASPIRE's LIBERO-PRO files, not stock LIBERO.
export LIBERO_CONFIG_PATH=$ROOT/libero_cfg MUJOCO_GL=egl PYTHONPATH=$OPENPI/third_party/libero:$VL/scripts
for SPLIT in train val; do
    R=$(cfg "c['rollouts']['repeats']['$SPLIT']")
    [ "$SPLIT" = train ] && R=${REPEATS_TRAIN:-$R} || R=${REPEATS_VAL:-$R}
    [ "$R" -gt 0 ] || continue
    STATES=$(cfg "' '.join(map(str, c['split']['$SPLIT'][:${MAX_STATES:-50}]))")
    mkdir -p "$OUT/$SPLIT"
    echo "$SPLIT: $R repeats x init states $STATES ($(date -Is))"
    $ROOT/.venv/bin/python $VL/scripts/libero_rollouts.py --args.port $PORT --args.task-suite-name $SUITE \
        --args.task-ids $TASK --args.init-states $STATES --args.repeats $R --args.seed $SEED \
        --args.save-raw --args.split-name $SPLIT --args.out-dir "$OUT/$SPLIT" 2>&1 | tee "$OUT/$SPLIT/client.log"
done
kill $SERVER 2>/dev/null || true
echo "rollouts done $(date -Is)"

for SPLIT in train val; do
    [ -f "$OUT/$SPLIT/episodes.csv" ] || continue
    if [ "$REPLAY" = all ]; then EPS=""; else
        EPS="--args.episodes $(seq -s ' ' 0 $((REPLAY - 1)))"; fi
    $ROOT/.venv/bin/python $VL/scripts/libero_replay.py --args.run-dir "$OUT/$SPLIT" $EPS 2>&1 \
        | tee "$OUT/$SPLIT/replay.log"
done
echo "end $(date -Is)"
