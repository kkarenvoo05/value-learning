"""scripts/libero_tag_usage.py for AdaRMS runs (scripts/pi05_tag_adarms.py) [ours, 2026-11-06].

Same tests and arguments; runs with an adarms.json are loaded with load_adarms (tag stripped from the prompt and fed
through adaRMS), every other run as before. The tests send tagged prompts, which the AdaRMS policy's input transform
parses, so the numbers are comparable with the prompt-route runs.

Usage (openpi env): .venv/bin/python <vl>/scripts/libero_tag_usage_adarms.py --out <dir> --run <adarms run> ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_tag_usage as tu  # noqa: E402
import pi05_tag_adarms as ta  # noqa: E402

_orig = tu.load_finetuned


def load(run_dir, step="final", seed=0):
    return ta.load_adarms(run_dir, step, seed=seed) if ta.is_adarms(run_dir) else _orig(run_dir, step, seed=seed)


tu.load_finetuned = load

if __name__ == "__main__":
    tu.main()
