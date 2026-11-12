"""Preemption resume of scripts/libero_rollouts.py [2026-11-07]: LIBERO, the policy client and imageio stubbed."""

import csv
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "libero_rollouts.py"


def load_module(monkeypatch):
    for name in ["imageio", "tyro", "libero", "libero.libero", "libero.libero.envs", "openpi_client",
                 "openpi_client.image_tools", "openpi_client.websocket_client_policy"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["imageio"].mimwrite = lambda *a, **k: None
    sys.modules["libero.libero"].benchmark = types.SimpleNamespace()
    sys.modules["libero.libero"].get_libero_path = lambda k: "."
    sys.modules["libero.libero.envs"].OffScreenRenderEnv = object
    sys.modules["openpi_client"].image_tools = sys.modules["openpi_client.image_tools"]
    sys.modules["openpi_client"].websocket_client_policy = sys.modules["openpi_client.websocket_client_policy"]
    spec = importlib.util.spec_from_file_location("libero_rollouts_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    suite = types.SimpleNamespace(n_tasks=1, get_task=lambda t: None, get_task_init_states=lambda t: list(range(4)))
    m.benchmark.get_benchmark_dict = lambda: {"libero_10": lambda: suite}
    m._get_libero_env = lambda task, res, seed: (types.SimpleNamespace(close=lambda: None), "task text")
    m._websocket_client_policy.WebsocketClientPolicy = lambda host, port: None
    return m


class Preempted(Exception):
    pass


def run(m, tmp_path, stop_after=None, resume=False):
    calls = []

    def fake_episode(env, client, init, desc, args, max_steps, noise_prefix=None, explore_rng=None):
        if stop_after is not None and len(calls) == stop_after:
            raise Preempted
        calls.append(noise_prefix)
        return init % 2 == 0, 100 + init, "", [], None

    m.run_episode = fake_episode
    args = m.Args(task_ids=[8], init_states=[0, 1, 2, 3], repeats=2, out_dir=str(tmp_path), matched_noise=True,
                  resume=resume)
    try:
        m.main(args)
    except Preempted:
        pass
    return calls


def rows(tmp_path):
    with open(tmp_path / "episodes.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_resume_runs_only_missing_episodes(monkeypatch, tmp_path):
    m = load_module(monkeypatch)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    first = run(m, tmp_path, stop_after=3)
    assert len(first) == 3 and len(rows(tmp_path)) == 3
    second = run(m, tmp_path, resume=True)
    assert second == [(8, i, r) for r, i in [(0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]]
    out = rows(tmp_path)
    assert [int(r["episode"]) for r in out] == list(range(8))  # plan positions, each once
    assert sorted((int(r["init_state"]), int(r["repeat"])) for r in out) == [(i, r) for i in range(4) for r in (0, 1)]


def test_requeue_env_resumes_without_flag_and_fresh_run_overwrites(monkeypatch, tmp_path):
    m = load_module(monkeypatch)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    run(m, tmp_path, stop_after=5)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    assert len(run(m, tmp_path)) == 3  # requeued job: resumes on its own
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    assert len(run(m, tmp_path)) == 8  # not requeued, no flag: starts over (old behaviour)
    assert len(rows(tmp_path)) == 8


def test_truncated_last_row_is_rerun(monkeypatch, tmp_path):
    m = load_module(monkeypatch)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    run(m, tmp_path, stop_after=4)
    text = (tmp_path / "episodes.csv").read_text().rstrip("\n")
    (tmp_path / "episodes.csv").write_text(text[: text.rfind(",") - 3])  # cut the last row mid-write
    assert len(m.load_done(tmp_path / "episodes.csv")) == 3
    assert len(run(m, tmp_path, resume=True)) == 5


# exploration noise [2026-11-08, data-variety pilot]
import json  # noqa: E402

import numpy as np  # noqa: E402


class StubEnv:
    """Minimal LIBERO env: fixed observations, done after `length` policy steps."""

    def __init__(self, length=23):
        self.length, self.n = length, 0
        self.env = types.SimpleNamespace(_eval_predicate=lambda g: False, parsed_problem={"goal_state": [1, 2]})

    def _obs(self):
        z = np.zeros((4, 4, 3), np.uint8)
        return {"agentview_image": z, "robot0_eye_in_hand_image": z, "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]), "robot0_gripper_qpos": np.zeros(2),
                "pot_pos": np.zeros(3)}

    def reset(self):
        self.n = 0
        return self._obs()

    def set_init_state(self, s):
        return self._obs()

    def get_sim_state(self):
        return np.zeros(2)

    def step(self, action):
        self.n += 1
        return self._obs(), 0.0, self.n >= 10 + self.length, {}


class StubClient:
    def __init__(self):
        self.requests = []

    def infer(self, element):
        self.requests.append(dict(element))
        return {"actions": np.zeros((10, 7))}


def episode(m, prob, scale=2.0, seed=0, save_raw=True):
    m.image_tools.resize_with_pad = lambda img, h, w: img
    m.image_tools.convert_to_uint8 = lambda img: img
    args = m.Args(save_raw=save_raw, explore_prob=prob, explore_scale=scale)
    client = StubClient()
    rng = np.random.default_rng(seed) if prob > 0 else None
    out = m.run_episode(StubEnv(), client, 0, "task", args, 520, explore_rng=rng)
    return client.requests, out[4]


def test_explore_sends_scale_on_drawn_chunks_and_records_them(monkeypatch):
    m = load_module(monkeypatch)
    req, raw = episode(m, 0.5)
    sent = [r.get("noise_scale") for r in req]
    assert len(req) == 5  # 23 steps, a new chunk every 5
    assert set(sent) <= {None, 2.0} and 0 < sent.count(2.0) < len(req)
    expect = np.random.default_rng(0).random(len(req)) < 0.5
    assert [s == 2.0 for s in sent] == expect.tolist()
    per_step = np.repeat(expect, 5)[:23]
    assert raw["noisy"].tolist() == per_step.tolist() and raw["new_chunk"].sum() == len(req)


def test_explore_off_is_unchanged(monkeypatch):
    m = load_module(monkeypatch)
    req, raw = episode(m, 0.0)
    assert all("noise_scale" not in r for r in req) and "noisy" not in raw
    req1, _ = episode(m, 1.0, scale=1.5)
    assert all(r["noise_scale"] == 1.5 for r in req1)


def test_explore_main_writes_settings_and_refuses_matched_noise(monkeypatch, tmp_path):
    m = load_module(monkeypatch)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    seen = []
    m.run_episode = lambda *a, explore_rng=None, **k: (seen.append(explore_rng.random()), (True, 5, "", [], None))[1]
    args = m.Args(task_ids=[8], init_states=[0, 1], repeats=1, out_dir=str(tmp_path), explore_prob=0.5,
                  explore_scale=2.0, explore_seed=3)
    m.main(args)
    assert json.loads((tmp_path / "explore.json").read_text()) == {"prob": 0.5, "scale": 2.0, "seed": 3}
    again = []
    m.run_episode = lambda *a, explore_rng=None, **k: (again.append(explore_rng.random()), (True, 5, "", [], None))[1]
    m.main(args)
    assert seen == again and seen[0] != seen[1]  # reproducible per episode, different across episodes
    with pytest.raises(AssertionError):
        m.main(m.Args(task_ids=[8], init_states=[0], out_dir=str(tmp_path), explore_prob=0.5, matched_noise=True))
