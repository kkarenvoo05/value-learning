import numpy as np

from vl import libero_finetune_data as fd
from vl import tag_route as tr

P = "put both moka pots on the stove"


def test_split_inverts_tag_prompt():
    for tag in (True, False, None):
        assert tr.split_tag(fd.tag_prompt(P, tag)) == (P, tag)


def test_split_leaves_other_prompts():
    assert tr.split_tag("Advantage: positive") == ("Advantage: positive", None)  # no ". " before it
    assert tr.split_tag(P + ". Advantage: positive!") == (P + ". Advantage: positive!", None)


def test_wrapped_transform_strips_and_masks():
    seen = []

    def base(d):
        seen.append(d["prompt"])
        return {"tokenized_prompt": d["prompt"], "image_mask": {"base_0_rgb": np.True_}}

    f = tr.wrap_input_transform(base)
    for tag in (True, False, None):
        out = f({"prompt": fd.tag_prompt(P, tag)})
        assert out["tokenized_prompt"] == P
        assert bool(out["image_mask"][tr.MASK_POS]) == (tag is True)
        assert bool(out["image_mask"][tr.MASK_NEG]) == (tag is False)
        assert out["image_mask"]["base_0_rgb"]
    assert seen == [P] * 3
