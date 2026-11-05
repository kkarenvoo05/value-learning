"""E2 subtask label (vl.libero_subtask) and ValueModel's auxiliary head."""

import numpy as np
import pytest
import torch

from vl.libero_subtask import N_SUBTASK, subtask_labels
from vl.model import ValueModel


def test_labels_shift_by_one_and_count_both_pots():
    # goals[t] is evaluated after action t: pot 1 lands after action 2, pot 2 after action 4, pot 1 leaves after 5
    g = np.zeros((7, 3), bool)
    g[2:5, 0] = True
    g[4:, 1] = True
    g[:, 2] = True  # third predicate ignored
    np.testing.assert_array_equal(subtask_labels(g), [0, 0, 0, 1, 1, 2, 1])
    assert subtask_labels(g).dtype == np.int64


def test_label_zero_frame_uses_first_predicate():
    g = np.ones((3, 2), bool)
    np.testing.assert_array_equal(subtask_labels(g), [2, 2, 2])


def test_aux_head_shapes_and_shared_hidden():
    torch.manual_seed(0)
    m = ValueModel(["agentview", "wrist"], pretrained=False, aux_classes=N_SUBTASK).eval()
    imgs = {c: torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8) for c in m.cams}
    with torch.no_grad():
        v = m(imgs)
        v2, aux = m(imgs, return_aux=True)
    assert v.shape == (2, 201) and aux.shape == (2, N_SUBTASK)
    torch.testing.assert_close(v, v2)  # return_aux leaves the value logits unchanged


def test_aux_off_by_default():
    m = ValueModel(["agentview"], pretrained=False)
    assert not hasattr(m, "aux_head")
    with pytest.raises(ValueError):
        m({"agentview": torch.zeros(1, 3, 64, 64, dtype=torch.uint8)}, return_aux=True)
