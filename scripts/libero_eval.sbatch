#!/bin/bash
# LIBERO eval of pi05_libero (docs/libero_plan.md Stage 0): openpi policy server + LIBERO client on one node.
# Usage: SUITE=libero_spatial TRIALS=1 NAME=smoke sbatch scripts/libero_eval.sbatch               (openpi main.py)
#        SUITE=libero_10 TASKS=8 REPEATS=2 SEED=8 NAME=moka sbatch scripts/libero_eval.sbatch  (scripts/libero_rollouts.py)
#        CLIENT_ARGS="--args.fresh-layouts 100 --args.save-raw" ...   (extra libero_rollouts.py arguments)
#        FT_RUN=<runs/policy_*> FT_TAG=positive|negative|none ...  Stage 6: serve a fine-tuned run
#        (scripts/serve_pi05_finetuned.py; FT_TAG default auto = positive for arm C, none otherwise)
#        FT_STEP=<n|final> serve trainable_<n>.msgpack (runs trained with --save-every)
#        FT_GUIDANCE=<beta> classifier-free guidance toward the positive tag (runs trained with --tag-dropout)
#        FT_SUBTASK=1 the client sends the moka subtask (runs trained with --subtask-prompt)
#        DSRL_MODE=train|eval [DSRL_AGENT=<agent_*.pkl>] [DSRL_SEED] [DSRL_ARGS] DSRL noise-space SAC on base pi05
#        (scripts/dsrl_pi05.py; train writes checkpoints + dsrl_log.jsonl to the run dir)
#        MATCHED_NOISE=1 [NOISE_SEED=0] same start noise per (task, init, repeat, chunk) in every eval
#        (TASKS mode only; serves base and fine-tuned runs through serve_pi05_finetuned.py; vl/libero_noise.py)
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris[1-4,9],iris-hgx-[1-2],iris-hp-z8   # Turing GPUs fail the pi05 bf16 self-check; iris9: JAX finds no CUDA device (2026-11-05); hgx: shared H100/H200s not needed, evals run on 24 GB cards (2026-11-08)
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
[ -n "${FT_SUBTASK:-}" ] && CLIENT_ARGS="${CLIENT_ARGS:-} --args.send-subtask"  # runs trained with --subtask-prompt
if [ -n "${MATCHED_NOISE:-}" ]; then  # matched noise needs our server (base too) and the keyed client
    CLIENT_ARGS="${CLIENT_ARGS:-} --args.matched-noise"; [ -n "${FT_RUN:-}" ] || BASE_OURS=1
    echo "matched noise: on (noise seed ${NOISE_SEED:-0})"
fi
if [ -n "${DSRL_MODE:-}" ]; then  # [2026-11-10] DSRL on base pi05 (scripts/dsrl_pi05.py), TASKS mode only
    [ -n "${TASKS:-}" ] && [ -z "${MATCHED_NOISE:-}" ] || { echo "DSRL needs TASKS and no MATCHED_NOISE"; exit 1; }
    echo "DSRL $DSRL_MODE agent ${DSRL_AGENT:-fresh} args ${DSRL_ARGS:-}"
    CLIENT_ARGS="${CLIENT_ARGS:-} --args.dsrl"
    .venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/dsrl_pi05.py --mode $DSRL_MODE \
        ${DSRL_AGENT:+--agent $DSRL_AGENT} --out "$OUT" --seed ${DSRL_SEED:-0} ${DSRL_ARGS:-} --port $PORT \
        > "$OUT/server.log" 2>&1 &
elif [ -n "${FT_RUN:-}" ] || [ -n "${BASE_OURS:-}" ]; then
    echo "fine-tuned run ${FT_RUN:-none (base, our server)} tag ${FT_TAG:-auto} step ${FT_STEP:-final}"
    .venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/serve_pi05_finetuned.py \
        $([ -n "${FT_RUN:-}" ] && echo "--run $FT_RUN" || echo "--base") --noise-seed ${NOISE_SEED:-0} \
        --tag ${FT_TAG:-auto} --step ${FT_STEP:-final} ${FT_GUIDANCE:+--guidance $FT_GUIDANCE} --port $PORT > "$OUT/server.log" 2>&1 &
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
        --args.seed $SEED --args.out-dir "$OUT" --args.resume ${CLIENT_ARGS:-} 2>&1 | tee -a "$OUT/client.log"
else                          # openpi's stock client: every task, TRIALS trials per task
    $ROOT/.venv/bin/python examples/libero/main.py --args.host 127.0.0.1 --args.port $PORT \
        --args.task-suite-name $SUITE --args.num-trials-per-task $TRIALS --args.seed $SEED \
        --args.video-out-path "$OUT/videos" 2>&1 | tee "$OUT/client.log"
fi
echo "end $(date -Is)"
