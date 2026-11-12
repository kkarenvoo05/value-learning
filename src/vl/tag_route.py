"""Advantage tag routed outside the prompt (scripts/pi05_tag_adarms.py) [ours, 2026-11-06].

The prompt route (vl.libero_finetune_data.tag_prompt) appends ". Advantage: positive|negative" to the task. The AdaRMS
route keeps every data path and tool unchanged (they still produce tagged prompts) and moves the tag at the policy's
input transform: the suffix is cut from the prompt before tokenisation and carried as two boolean entries of the
observation's image_mask dict (openpi's Observation keeps that dict as is; preprocess_observation and embed_prefix only
read the camera keys, so the extra entries reach the model's sample_actions / compute_loss and nothing else).
"""

from __future__ import annotations

from vl.libero_finetune_data import TAG_FORMAT, tag_prompt

MASK_POS, MASK_NEG = "adv_tag_pos", "adv_tag_neg"  # image_mask keys carrying the tag
_SUFFIX = {tag: TAG_FORMAT.format(prompt="", tag=word) for tag, word in ((True, "positive"), (False, "negative"))}


def split_tag(prompt: str) -> tuple[str, bool | None]:
    """Inverse of tag_prompt: (task prompt, True / False / None)."""
    for tag, suffix in _SUFFIX.items():
        if prompt.endswith(suffix):
            return prompt[: -len(suffix)], tag
    return prompt, None


def tag_masks(tag: bool | None) -> dict:
    """The two image_mask entries for one sample (none = both False)."""
    import numpy as np
    return {MASK_POS: np.bool_(tag is True), MASK_NEG: np.bool_(tag is False)}


def wrap_input_transform(transform):
    """Policy input transform that strips the tag from the prompt and adds it as image_mask entries."""
    def f(data: dict) -> dict:
        data = dict(data)
        prompt, tag = split_tag(str(data["prompt"]))
        data["prompt"] = prompt
        out = transform(data)
        out["image_mask"] = {**out["image_mask"], **tag_masks(tag)}
        return out
    return f

