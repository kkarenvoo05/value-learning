#!/bin/bash
# AdaRMS tag route (scripts/pi05_tag_adarms.py), openpi env, one GPU (48 GB: A40 / A6000 / L40S).
# Usage:
#   MODE=selftest sbatch scripts/pi05_tag_adarms.sbatch
#   MODE=train NAME=armC_adarms_vb40_s0 ARGS="--train --arm advantage --labels <parquet> --tag-dropout 0.3 --demos
#       --freeze-siglip --save-every 1000 --seed 0" sbatch scripts/pi05_tag_adarms.sbatch      (= faithful RECAP args)
#   MODE=tagusage ARGS="--run <run> --labels <parquet> --steps 1000 3000 final --arms tagged" sbatch ...
#   MODE=eval FT_RUN=<run dir> [FT_TAG=positive|negative|none] [FT_STEP=final|<n>] [FT_GUIDANCE=1.5] [MATCHED_NOISE=1
#       NOISE_SEED=0] NAME=eval_adarms sbatch scripts/pi05_tag_adarms.sbatch
#       FT_CONTRAST=<w>: contrastive guidance v_none + w (v_pos - v_neg) instead of a tag (Advantage Guidance Eq. 4)
#   PY_SCRIPT=pi05_ag_residual.py [2026-11-11]: the same modes (selftest -> --residual-selftest, train, eval) for the
#       Advantage Guidance residual parameterisation (frozen pi0.5 + residual head); contrastcheck works on its runs
#   MODE=contrastcheck ARGS="--run <run> --weights 1 2 4 8": offline self-check + action shift per w (one GPU)
#       (the 330 policy-eval trials: libero_10 task 8, the 10 val init states x 33 repeats, env seed 7, as
#        scripts/libero_eval.sbatch TASKS mode)
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris[1-4,8-10],iris-hgx-[1-2],iris-hp-z8   # 48 GB cards only; iris9-10: JAX finds no CUDA device (iris10 ECC errors 2026-11-11)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/iris/u/ankile/karenvo/libero/logs/slurm-%j.out
set -euo pipefail
: "${MODE:?MODE=selftest|train|eval}"
VL=/move/u/karenvo/Projects/value-learning
OPENPI=/iris/u/ankile/karenvo/pi05/openpi
ROOT=/iris/u/ankile/karenvo/libero
export UV_CACHE_DIR=/iris/u/ankile/karenvo/pi05/uv-cache XDG_CACHE_HOME=/iris/u/ankile/karenvo/pi05/cache
export OPENPI_DATA_HOME=/iris/u/ankile/karenvo/pi05/cache/openpi
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) start $(date -Is) mode $MODE"
cd $OPENPI
PY=$VL/scripts/${PY_SCRIPT:-pi05_tag_adarms.py}
SELFTEST_FLAG=$([ "${PY_SCRIPT:-}" = pi05_ag_residual.py ] && echo --residual-selftest || echo --adarms-selftest)
case $MODE in
selftest)
    .venv/bin/python $PY $SELFTEST_FLAG --out $ROOT/runs/${PY_SCRIPT:+residual_}adarms_selftest_${SLURM_JOB_ID}
    ;;
train)
    OUT=$ROOT/runs/policy_${NAME:-adarms}_${SLURM_JOB_ID}
    RESUME=""   # requeued job: warm restart from the newest checkpoint in its own OUT (as pi05_policy_finetune.sbatch)
    if [ -d "$OUT" ]; then
        last=$(ls "$OUT" | sed -n 's/^trainable_\([0-9]\+\)\.msgpack$/\1/p' | sort -n | tail -1)
        if [ -n "$last" ]; then
            ARGS=$(echo " $ARGS " | sed -E 's/ --resume-(from|step) [^ ]+//g')
            RESUME="--resume-from $OUT --resume-step $last"; echo "requeued: resuming from step $last"
        fi
    fi
    .venv/bin/python $PY ${TAG_SCALE:+--tag-scale $TAG_SCALE} $ARGS $RESUME --out $OUT
    ;;
tagusage)  # ARGS="--run <adarms run> --labels <parquet> --steps 1000 3000 final --arms tagged"
    OUT=${OUT:-$ROOT/runs/tag_usage_adarms_${SLURM_JOB_ID}}
    .venv/bin/python $VL/scripts/libero_tag_usage_adarms.py --out $OUT ${ARGS:-}
    ;;
contrastcheck)
    OUT=${OUT:-$ROOT/runs/contrast_check_${SLURM_JOB_ID}}
    .venv/bin/python $VL/scripts/pi05_contrast_check.py --out $OUT ${ARGS:-}
    ;;
eval)
    : "${FT_RUN:?set FT_RUN}"
    if [ -n "${FT_SHIFT_TARGET:-}" ]; then  # [2026-11-11] matched action shift: w = target / slope, the slope of
        # shift/noise vs w (least squares through 0) in the contrastcheck results.json FT_SHIFT_FROM
        FT_CONTRAST=$(python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['weights']; p=[(float(w),v['norm']['ratio']) for w,v in r.items() if float(w)>0]; k=sum(w*x for w,x in p)/sum(w*w for w,x in p); print(round(float(sys.argv[2])/k,3))" "${FT_SHIFT_FROM:?set FT_SHIFT_FROM}" $FT_SHIFT_TARGET)
        echo "matched shift target $FT_SHIFT_TARGET -> contrast w $FT_CONTRAST"
    fi
    OUT=$ROOT/runs/${NAME:-eval_adarms}_libero_10_${SLURM_JOB_ID}
    PORT=$((8100 + SLURM_JOB_ID % 800))
    mkdir -p "$OUT"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
    # default = the 330 policy-eval trials; CLIENT_ARGS / REPEATS / SEED override it (RECAP round-2 collection)
    CLIENT_ARGS="${CLIENT_ARGS:---args.init-states 9 12 13 17 25 32 40 42 44 48}"
    [ -n "${FT_SUBTASK:-}" ] && CLIENT_ARGS="$CLIENT_ARGS --args.send-subtask"
    [ -n "${MATCHED_NOISE:-}" ] && CLIENT_ARGS="$CLIENT_ARGS --args.matched-noise"
    echo "run $FT_RUN step ${FT_STEP:-final} tag ${FT_TAG:-positive} guidance ${FT_GUIDANCE:-none} contrast ${FT_CONTRAST:-none} noise seed ${NOISE_SEED:-0} out $OUT"
    .venv/bin/python $PY --serve --run $FT_RUN --step ${FT_STEP:-final} \
        --tag ${FT_TAG:-positive} ${FT_GUIDANCE:+--guidance $FT_GUIDANCE} ${FT_CONTRAST:+--contrast $FT_CONTRAST} --noise-seed ${NOISE_SEED:-0} \
        --port $PORT > "$OUT/server.log" 2>&1 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null || true' EXIT
    for i in $(seq 1 120); do
        if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
        kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -50 "$OUT/server.log"; exit 1; }
        sleep 5
    done
    echo "server up $(date -Is)"
    export LIBERO_CONFIG_PATH=$ROOT/libero_cfg MUJOCO_GL=egl PYTHONPATH=$OPENPI/third_party/libero
    $ROOT/.venv/bin/python $VL/scripts/libero_rollouts.py --args.port $PORT --args.task-suite-name libero_10 \
        --args.task-ids 8 --args.repeats ${REPEATS:-33} --args.seed ${SEED:-7} --args.out-dir "$OUT" --args.resume $CLIENT_ARGS 2>&1 | tee -a "$OUT/client.log"
    ;;
esac
echo "end $(date -Is)"
