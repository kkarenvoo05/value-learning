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
#       FT_ANCHOR=<a> [2026-11-17, residual only]: guide from the data's behaviour, mix (a + w, 1 - a - w, 0)
#       (pi05_ag_residual.anchored_mix); w still comes from FT_CONTRAST / FT_SHIFT_TARGET
#       FT_UNCOND=1 [2026-11-17, residual runs trained with --uncond-branch]: v_ref + h(0) + w (h(+) - h(-))
#       FT_CONTRAST_SUB1=<w> / FT_SHIFT_TARGET_SUB1=<shift> [2026-11-12, residual only]: a second w once a moka pot is
#       on the stove (client --args.send-subtask, server --contrast-sub1)
#   PY_SCRIPT=pi05_ag_residual.py [2026-11-11]: the same modes (selftest -> --residual-selftest, train, eval) for the
#       Advantage Guidance residual parameterisation (frozen pi0.5 + residual head); contrastcheck works on its runs
#   MODE=contrastcheck ARGS="--run <run> --weights 1 2 4 8": offline self-check + action shift per w (one GPU)
#       (the 330 policy-eval trials: libero_10 task 8, the 10 val init states x 33 repeats, env seed 7, as
#        scripts/libero_eval.sbatch TASKS mode)
#SBATCH --job-name=job
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris[1-4,7-10],iris-hgx-[1-2],iris-hp-z8   # 48 GB cards only; iris9-10: JAX finds no CUDA device (iris10 ECC errors 2026-11-11)
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
pushdir)  # [2026-11-21] PUSH_LIST=<file of "name run check" lines>, OUT=<dir>: one process per run (frees memory),
    # scripts/pi05_push_direction.py -> <OUT>/<name>/push.npz; runs already done are skipped (requeue-safe)
    : "${PUSH_LIST:?set PUSH_LIST}" "${OUT:?set OUT}"
    while read -r name run chk; do
        [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
        [ -e "$OUT/$name/push.npz" ] && { echo "skip $name (done)"; continue; }
        .venv/bin/python $VL/scripts/pi05_push_direction.py --out $OUT/$name --run $run --check $chk ${ARGS:-}
    done < "$PUSH_LIST"
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
    shift_to_w() {  # [2026-11-11] matched action shift: w = target / slope, the slope of shift/noise vs w (least
        # squares through 0) in the contrastcheck results.json FT_SHIFT_FROM
        python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['weights']; p=[(float(w),v['norm']['ratio']) for w,v in r.items() if float(w)>0]; k=sum(w*x for w,x in p)/sum(w*w for w,x in p); print(round(float(sys.argv[2])/k,3))" "${FT_SHIFT_FROM:?set FT_SHIFT_FROM}" "$1"
    }
    shift_to_w_interp() {  # [2026-11-19] FT_SHIFT_INTERP=1: w where the measured shift/noise curve (piecewise linear
        # through (0, 0) and the check's weights) reaches the target; linear extrapolation past the last weight. For
        # checks whose shift grows faster than linearly in w (the task-64 demo runs: 0.34 / 0.94 / 2.6 / 7.6 / 23.6 at
        # w 1-16), where the least-squares slope is set by w = 16 and gives ~1/4 of the target push. Default unchanged.
        python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))['weights']; t = float(sys.argv[2])
p = [(0.0, 0.0)] + sorted((float(w), v['norm']['ratio']) for w, v in r.items() if float(w) > 0)
assert all(b[1] > a[1] for a, b in zip(p, p[1:])), p
seg = next(((a, b) for a, b in zip(p, p[1:]) if b[1] >= t), (p[-2], p[-1]))
(w0, x0), (w1, x1) = seg
print(round(w0 + (t - x0) * (w1 - w0) / (x1 - x0), 3))" "${FT_SHIFT_FROM:?set FT_SHIFT_FROM}" "$1"
    }
    if [ -n "${FT_SHIFT_TARGET:-}" ] && [ -n "${FT_SHIFT_INTERP:-}" ]; then
        FT_CONTRAST=$(shift_to_w_interp $FT_SHIFT_TARGET)
        echo "matched shift target $FT_SHIFT_TARGET -> contrast w $FT_CONTRAST (interpolated)"
    elif [ -n "${FT_SHIFT_TARGET:-}" ]; then
        FT_CONTRAST=$(shift_to_w $FT_SHIFT_TARGET)
        echo "matched shift target $FT_SHIFT_TARGET -> contrast w $FT_CONTRAST"
    fi
    # [2026-11-18] PY_SCRIPT=pi05_ag_residual.py only: FT_SHIFT_SET="<push> ..." = mixed-push collection, each episode
    # served at one push of the set (client --args.push-choices, seed PUSH_SEED; server --contrast-set, same slope).
    # Separator ":" or space (":" survives an unquoted --export list, e.g. FT_SHIFT_SET=0.5:0.99:1.5:2.0).
    if [ -n "${FT_SHIFT_SET:-}" ]; then
        FT_SHIFT_SET=${FT_SHIFT_SET//:/ }
        [ "${PY_SCRIPT:-}" = pi05_ag_residual.py ] && [ -z "${MATCHED_NOISE:-}" ] ||
            { echo "FT_SHIFT_SET needs PY_SCRIPT=pi05_ag_residual.py and no MATCHED_NOISE"; exit 1; }
        FT_CONTRAST_SET=$(for x in $FT_SHIFT_SET; do shift_to_w $x; done | tr '\n' ' ')
        FT_CONTRAST=${FT_CONTRAST:-${FT_CONTRAST_SET%% *}}
        PUSH_CLIENT="--args.push-choices $(echo $FT_SHIFT_SET | wc -w) --args.push-seed ${PUSH_SEED:-0}"
        echo "mixed-push set $FT_SHIFT_SET -> contrast w $FT_CONTRAST_SET"
    fi
    # [2026-11-12] PY_SCRIPT=pi05_ag_residual.py only: a second contrast w once a pot is on the stove (subtask 1),
    # FT_CONTRAST_SUB1=<w> or FT_SHIFT_TARGET_SUB1=<shift> (same slope as above); the client then sends the subtask
    if [ -n "${FT_SHIFT_TARGET_SUB1:-}" ]; then
        FT_CONTRAST_SUB1=$(shift_to_w $FT_SHIFT_TARGET_SUB1)
        echo "subtask-1 shift target $FT_SHIFT_TARGET_SUB1 -> contrast w $FT_CONTRAST_SUB1"
    fi
    if [ -n "${FT_CONTRAST_SUB1:-}" ]; then
        [ "${PY_SCRIPT:-}" = pi05_ag_residual.py ] && [ -n "${FT_CONTRAST:-}" ] ||
            { echo "FT_CONTRAST_SUB1 needs PY_SCRIPT=pi05_ag_residual.py and FT_CONTRAST / FT_SHIFT_TARGET"; exit 1; }
        FT_SUBTASK=1
    fi
    # [2026-11-13] residual heads with a subtask input (residual.json head.subtask): the client must send the subtask
    if [ -f "$FT_RUN/residual.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('head', {}).get('subtask') else 1)" "$FT_RUN/residual.json"; then
        FT_SUBTASK=1; echo "head takes the subtask: client sends it"
    fi
    # [2026-11-16, multi-task AG] AG_SUITE / AG_TASK (default libero_10 / 8 = moka) pick the task for collect / eval;
    # AG_MAX_STEPS > 0 overrides the suite's step cap (520 for libero_90 task 64, as moka). Distinct names from
    # libero_eval.sbatch's SUITE / TASKS so an exported SUITE never redirects these jobs.
    AG_SUITE=${AG_SUITE:-libero_10} AG_TASK=${AG_TASK:-8}
    OUT=$ROOT/runs/${NAME:-eval_adarms}_${AG_SUITE}_${SLURM_JOB_ID}
    PORT=$((8100 + SLURM_JOB_ID % 800))
    mkdir -p "$OUT"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
    # default = the 330 policy-eval trials; CLIENT_ARGS / REPEATS / SEED override it (RECAP round-2 collection)
    CLIENT_ARGS="${CLIENT_ARGS:---args.init-states 9 12 13 17 25 32 40 42 44 48}"
    [ -n "${FT_SUBTASK:-}" ] && CLIENT_ARGS="$CLIENT_ARGS --args.send-subtask"
    [ -n "${MATCHED_NOISE:-}" ] && CLIENT_ARGS="$CLIENT_ARGS --args.matched-noise"
    [ -n "${PUSH_CLIENT:-}" ] && CLIENT_ARGS="$CLIENT_ARGS $PUSH_CLIENT"
    echo "run $FT_RUN step ${FT_STEP:-final} tag ${FT_TAG:-positive} guidance ${FT_GUIDANCE:-none} contrast ${FT_CONTRAST:-none} contrast-sub1 ${FT_CONTRAST_SUB1:-none} anchor ${FT_ANCHOR:-none} uncond ${FT_UNCOND:-0} noise seed ${NOISE_SEED:-0} out $OUT"
    .venv/bin/python $PY --serve --run $FT_RUN --step ${FT_STEP:-final} \
        --tag ${FT_TAG:-positive} ${FT_GUIDANCE:+--guidance $FT_GUIDANCE} ${FT_CONTRAST:+--contrast $FT_CONTRAST} \
        ${FT_CONTRAST_SUB1:+--contrast-sub1 $FT_CONTRAST_SUB1} ${FT_CONTRAST_SET:+--contrast-set $FT_CONTRAST_SET} ${FT_ANCHOR:+--anchor $FT_ANCHOR} ${FT_UNCOND:+--uncond} --noise-seed ${NOISE_SEED:-0} \
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
    $ROOT/.venv/bin/python $VL/scripts/libero_rollouts.py --args.port $PORT --args.task-suite-name $AG_SUITE \
        --args.task-ids $AG_TASK ${AG_MAX_STEPS:+--args.max-steps $AG_MAX_STEPS} --args.repeats ${REPEATS:-33} --args.seed ${SEED:-7} --args.out-dir "$OUT" --args.resume $CLIENT_ARGS 2>&1 | tee -a "$OUT/client.log"
    ;;
esac
echo "end $(date -Is)"
