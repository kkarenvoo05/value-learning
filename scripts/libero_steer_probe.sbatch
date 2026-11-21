#!/bin/bash
# Best-of-N steering pre-checks (docs/CURRENT.md B2 steps 1-2): base pi05_libero server + scripts/libero_steer_probe.py.
# Usage: NAME=steer_probe sbatch --dependency=afterany:<job> scripts/libero_steer_probe.sbatch
#        CLIENT_ARGS="--args.n-cand 8 --args.repeats 2" ...  (extra libero_steer_probe.py arguments)
#        SCALE=1.5 ...  v2: candidates 1..N-1 with start noise x SCALE (serves via serve_pi05_finetuned.py --base,
#                       which honours a per-request noise_scale; same pi05_libero weights and transforms)
#        DSRL_AGENT=<agent_*.pkl> [DSRL_SEED=0] CLIENT_ARGS="--args.cand-source actor|gauss ..."  [2026-11-13]
#                       DSRL G1 probe: the server is scripts/dsrl_pi05.py --mode eval (frozen pi05_libero + the agent);
#                       trunk and continuations = the DSRL policy; scored by scripts/libero_dsrl_g1.py
#        CLIENT_ARGS="--args.cand-offsets none x+ x- y+ y- z+ z- z+0.5 --args.hold-chunks H ..."  [2026-11-20]
#                       structured-offset G0: plain pi05_libero server; candidates = one chunk + end-effector offsets
#                       (libero_steer_probe.py docstring); resumes after a requeue (same job id -> same $OUT)
#        SOURCES="actor gauss" (with DSRL_AGENT): one client run per candidate source, in turn, into $OUT/<source>/
#                       against the same server (one queue slot for both probes)
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8   # Turing GPUs fail the pi05 bf16 self-check; iris8-10 bad GPUs (as ag_rounds.sh, 2026-11-13); iris7 hung 2026-11-20; hgx not needed
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
if [ -n "${DSRL_AGENT:-}" ]; then
    echo "DSRL probe server, agent $DSRL_AGENT seed ${DSRL_SEED:-0}"
    .venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/dsrl_pi05.py --mode eval --agent "$DSRL_AGENT" \
        --seed ${DSRL_SEED:-0} --port $PORT > "$OUT/server.log" 2>&1 &
elif [ -n "${SCALE:-}" ]; then
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
if [ -n "${SOURCES:-}" ]; then
    for SRC in $SOURCES; do
        mkdir -p "$OUT/$SRC"
        echo "client: candidates $SRC $(date -Is)"
        $ROOT/.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/libero_steer_probe.py \
            --args.port $PORT --args.task-ids 8 --args.init-states $INIT --args.seed $SEED --args.out-dir "$OUT/$SRC" \
            --args.cand-source $SRC ${CLIENT_ARGS:-} 2>&1 | tee "$OUT/$SRC/client.log"
    done
else
$ROOT/.venv/bin/python /move/u/karenvo/Projects/value-learning/scripts/libero_steer_probe.py \
    --args.port $PORT --args.task-ids 8 --args.init-states $INIT --args.seed $SEED --args.out-dir "$OUT" \
    ${CLIENT_ARGS:-} 2>&1 | tee "$OUT/client.log"
fi
echo "end $(date -Is)"
