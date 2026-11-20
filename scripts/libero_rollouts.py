"""LIBERO rollouts of a pi05 policy server (docs/libero_plan.md Stage 1+), one video and one CSV row per episode.

Runs in the LIBERO client env (Python 3.8), against an openpi policy server; see scripts/libero_eval.sbatch.
The episode loop is a copy of openpi's examples/libero/main.py (215abfb), kept behaviour-identical (wait steps,
180-degree image rotation, resize-with-pad to 224, replan every 5 steps, max_steps per suite). Changes [ours]:
  1. --args.task-ids: run only these tasks of the suite (default: all)
  2. one video per episode, named by episode index, init state and outcome (main.py overwrote repeats)
  3. episodes.csv: one row per episode (task, init state, repeat, env seed, success, policy steps, error)
  4. (Stage 2) --args.save-raw: one raw/epNNNN.npz per episode with everything the policy saw and did, for the
     LeRobot conversion (scripts/libero_to_lerobot.py) and the replay check (scripts/libero_replay.py):
     image / wrist_image (T, 224, 224, 3) uint8 exactly as sent to the server, state (T, 8), executed actions
     (T, 7) (both in their original dtype), new_chunk (T,) whether step t started a new action chunk, goal predicates (T, G) after each step,
     object positions (T, 3 * n_obj) after each step, MuJoCo states: init (as stored by LIBERO), start (after
     the wait steps, before the first policy action) and end. Row t pairs the observation with the action
     taken from it. --args.split-name labels the rows (train / val)
Episode order: for each repeat r, for each init state i -> init state i, so every init state gets `repeats`
trials. Policy sampling noise is not seeded per episode by default: the server draws it from jax.random.key(0),
split on every infer call, so it depends on call order within a server's lifetime.
  5. --args.matched-noise [ours, 2026-11-06]: every request carries noise_key = [task_id, init_state, repeat, chunk
     index within the episode]; the server (scripts/serve_pi05_finetuned.py, not openpi's serve_policy.py) draws the
     start noise from that key (vl.libero_noise), so the same episode gets the same noise in every eval.
  6. --args.explore-prob p --args.explore-scale s [ours, 2026-11-08, data-variety pilot]: each new action chunk is,
     with probability p, requested with "noise_scale" = s (the server draws the flow-matching start noise as
     s * N(0, I); scripts/serve_pi05_finetuned.py, also for --base), otherwise as usual. The choice is drawn from
     an rng seeded by (explore_seed, task_id, init_state, repeat), so it is reproducible and a resumed episode
     gets the same choices. raw npz gains `noisy` (T,) = whether step t's chunk was a scaled one, and the run
     dir gets explore.json. Not with matched noise.
  7. --args.start-from <starts.csv> [ours, 2026-11-14, mid-task starts]: one episode per row instead of the init-state
     plan. Each row names a saved train rollout (source_raw) and a step k: set its sim_init, the wait steps, then
     replay its first k executed actions open-loop (exact: scripts/libero_replay.py), check the 8-d state equals the
     source's state[k], and hand over to the policy for the remaining max_steps - k steps (same time limit as a full
     episode). Only the policy's part is recorded; raw adds start_step, prefix_actions, source_raw, sim_wait (state
     after the wait steps) and sim_start = the state at hand-over. CSV: init_state = the source's, repeat =
     start_id, max_steps = max_steps - k, plus start_step / source_raw / prefix_maxdiff. The env is re-created with
     each source's env seed. Not with fresh layouts or matched noise.
  8. [2026-11-14, distillation] with --args.save-raw, raw also holds every sampled chunk: chunk_actions (m, H, 7) =
     the full chunk the server returned (H = 10; only the first replan_steps are executed) and chunk_t (m,) = the
     policy step it was requested at (= the frames where new_chunk is true), so a fine-tune can train on the
     policy's own chunks instead of the executed actions, which join two samples (vl.libero_finetune_data
     CHUNK_TARGETS). --args.dsrl-base-prefix a (PLD-style hybrid rollouts, arXiv 2511.00091 Sec. 3.2; needs a
     DSRL eval server, --args.dsrl): each episode draws a switch step k ~ U[0, a * max_steps) (rng seeded by
     dsrl_prefix_seed, task, init state, repeat, episode); chunks requested before step k carry dsrl_cand = "base"
     (plain pi0.5), later ones come from the DSRL actor. raw adds switch_step and chunk_actor (m,) 0/1; the CSV
     row is unchanged (the log line names k). a = 0 is a pure-actor rollout with the same bookkeeping.
  9. --args.push-choices K --args.push-seed s [ours, 2026-11-18, mixed-push collection]: each episode draws a
     guidance arm w_idx in [0, K) (rng seeded by push_seed, task, init state + 1, repeat, episode) and sends it as
     obs["w_idx"] with every request; the AG server's --contrast-set w_0 .. w_{K-1} serves the episode at w_idx's
     w. raw adds w_idx; the CSV gains a w_idx column.

Usage:
    python scripts/libero_rollouts.py --args.task-suite-name libero_10 --args.task-ids 3 \
        --args.repeats 2 --args.seed 8 --args.out-dir <run dir> --args.port <port>
"""

import collections
import csv
import dataclasses
import logging
import math
import os
import pathlib
from typing import List, Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
CSV_FIELDS = ["episode", "suite", "task_id", "task", "split", "init_state", "repeat", "env_seed", "success",
              "timeout", "policy_steps", "max_steps", "goals_end", "error", "video", "raw"]
START_FIELDS = ["start_step", "source_raw", "prefix_maxdiff"]  # Changes 7 (mid-task starts) only
PUSH_FIELDS = ["w_idx"]  # Changes 9 (per-episode guidance arm) only


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_10"
    task_ids: Optional[List[int]] = None  # default: all tasks of the suite
    init_states: Optional[List[int]] = None  # default: all init states of each task (50)
    repeats: int = 1  # trials per init state
    num_steps_wait: int = 10  # do-nothing steps while objects settle
    out_dir: str = "data/libero/run"
    seed: int = 7  # env seed (as main.py); not the policy's noise
    save_raw: bool = False  # write raw/epNNNN.npz per episode (Stage 2)
    split_name: str = ""  # written to the CSV split column
    max_steps: int = 0  # [2026-11-16] > 0 overrides the per-suite MAX_STEPS (e.g. 520 for libero_90, as libero_10)
    # > 0: this many episodes per task, each from a fresh random layout (env.reset() samples the task's BDDL placement
    # regions, as the 50 saved init states were drawn) instead of the saved init states; init_states / repeats are
    # ignored, the CSV init_state is -1, and raw sim_init holds the sampled state (replay: set_init_state(sim_init))
    fresh_layouts: int = 0
    # send noise_key per request so the server draws matched (common-random-numbers) noise; needs our server
    matched_noise: bool = False
    # [2026-11-07] preemption-safe: keep the complete rows of an existing episodes.csv in out_dir and run only the
    # missing (task, init state, repeat) episodes, appending. Episode numbers stay the plan position. Not with
    # fresh_layouts (their layouts depend on the env RNG's history). Reruns are not bit-identical (GPU / render
    # nondeterminism: the same matched-noise episode took 383 vs 460 steps in two attempts of 18355679), so a
    # resumed eval is a valid draw of every episode, not a replay of the interrupted attempt.
    resume: bool = False
    # [2026-11-07] send obs["subtask"] = 0 / 1 (no moka pot on the stove yet / at least one, from the BDDL goal
    # predicates now, i.e. the state the image shows) for runs trained with --subtask-prompt (serve_pi05_finetuned)
    send_subtask: bool = False
    # [2026-11-08] exploration: share of chunks sampled with start noise x explore_scale (0 = off)
    explore_prob: float = 0.0
    explore_scale: float = 1.0
    explore_seed: int = 0
    # [2026-11-10] DSRL (scripts/dsrl_pi05.py): after each episode send {"dsrl_end": 1, "success": 0/1, final
    # images/state} so the server can store the episode and update its SAC agent (train mode; eval ignores it)
    dsrl: bool = False
    # [2026-11-14] mid-task starts (Changes 7): CSV with start_id, source_raw, start_step, init_state, env_seed
    start_from: str = ""
    # [2026-11-14] PLD-style hybrid rollouts (Changes 8): base pi0.5 for a random prefix of up to this share of
    # max_steps, then the DSRL actor; < 0 = off (every chunk from the server's default source)
    dsrl_base_prefix: float = -1.0
    dsrl_prefix_seed: int = 0
    # [2026-11-18] per-episode guidance arm (Changes 9): each episode draws w_idx uniformly in [0, push_choices) from
    # an rng seeded by (push_seed, task_id, init_state + 1, repeat, episode) and sends obs["w_idx"] on every request;
    # the AG server (pi05_ag_residual.py --contrast-set) serves that episode with the w_idx-th contrast w. 0 = off.
    push_choices: int = 0
    push_seed: int = 0


def goal_predicates(env):
    """Each conjunct of the task's BDDL goal, evaluated now (success = all of them)."""
    inner = env.env
    return np.array([bool(inner._eval_predicate(g)) for g in inner.parsed_problem["goal_state"]])


def episode_goals_end(env, raw, policy_steps):
    """goals_end CSV field: the goal predicates after the episode's last step as a 0/1 string ("" if no policy step).
    [2026-11-12] Read from the env for every episode, so evals without --args.save-raw also record per-stage outcomes
    (task 8: first two chars = the two moka pots on the stove). The env is still in its end state here (run_episode
    does not reset it), so this equals raw["goals"][-1] when raw is saved; if the predicates cannot be evaluated
    (episode ended on an exception), fall back to raw or ""."""
    if policy_steps <= 0:
        return ""
    try:
        return "".join(str(int(g)) for g in goal_predicates(env))
    except Exception as e:  # noqa: BLE001 - logging only; the episode row is still written
        logging.warning(f"goals_end unavailable: {e!r}")
        return "".join(str(int(g)) for g in raw["goals"][-1]) if raw is not None and len(raw.get("goals", [])) else ""


def object_pos_keys(obs):
    return sorted(k for k in obs if k.endswith("_pos") and not k.startswith("robot0"))


def replay_prefix(env, obs, prefix):
    """Mid-task start (Changes 7): step the source's first k executed actions; returns (obs, max |state - source
    state[k]|). The 8-d state is the client's (eef pos, axis-angle, gripper qpos)."""
    for a in prefix["actions"]:
        obs, _, done, _ = env.step(a.tolist())
        assert not done, "source episode succeeded inside the prefix"
    state = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return obs, float(np.abs(state - prefix["state_k"]).max())


def run_episode(env, client, init_state, task_description, args, max_steps, noise_prefix=None, explore_rng=None,
                prefix=None, switch_step=None, w_idx=None):
    """One episode, as main.py. Returns (success, policy_steps, error, frames, raw); raw is None unless save_raw.
    explore_rng: with args.explore_prob > 0, draws which chunks are requested with noise_scale = args.explore_scale.
    prefix: mid-task start (Changes 7), {"actions": (k, 7), "state_k": (8,), "source_raw": str}; max_steps is then
    the remaining budget; prefix["maxdiff"] is set to the hand-over state check.
    switch_step: hybrid rollout (Changes 8): chunks requested before this policy step use dsrl_cand "base".
    w_idx: per-episode guidance arm (Changes 9), sent as obs["w_idx"] with every request."""
    obs = env.reset()
    action_plan = collections.deque()
    if init_state is None:  # fresh layout: keep the placement reset() sampled, record it for replay
        init_state = env.get_sim_state()
    else:
        obs = env.set_init_state(init_state)
    t, policy_steps, done, error, frames, chunk = 0, 0, False, "", [], 0
    noisy = False
    rec = collections.defaultdict(list)
    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                if t == args.num_steps_wait:
                    sim_start, pos_keys = env.get_sim_state(), object_pos_keys(obs)
                    if prefix is not None:
                        sim_wait = sim_start
                        obs, prefix["maxdiff"] = replay_prefix(env, obs, prefix)
                        sim_start = env.get_sim_state()
                continue
            # rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
            frames.append(img)
            state = np.concatenate(
                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
            new_chunk = not action_plan
            if new_chunk:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": str(task_description),
                }
                if args.send_subtask:
                    element["subtask"] = int(min(int(goal_predicates(env)[:2].sum()), 1))
                if noise_prefix is not None:  # matched noise: [task_id, init_state, repeat, chunk]
                    element["noise_key"] = np.array([*noise_prefix, chunk], dtype=np.int64)
                noisy = explore_rng is not None and bool(explore_rng.random() < args.explore_prob)
                if noisy:
                    element["noise_scale"] = float(args.explore_scale)
                if w_idx is not None:  # Changes 9
                    element["w_idx"] = int(w_idx)
                actor = switch_step is None or policy_steps >= switch_step
                if not actor:  # Changes 8: base pi0.5 before the hand-over to the DSRL actor
                    element["dsrl_cand"] = "base"
                chunk += 1
                action_chunk = client.infer(element)["actions"]
                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[: args.replan_steps])
                if args.save_raw:  # Changes 8: the whole sampled chunk and the step it was requested at
                    rec["chunk_actions"].append(np.asarray(action_chunk))
                    rec["chunk_t"].append(policy_steps)
                    if switch_step is not None:
                        rec["chunk_actor"].append(actor)
            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            policy_steps += 1
            if args.save_raw:
                rec["image"].append(img)
                rec["wrist_image"].append(wrist_img)
                rec["state"].append(state)
                rec["actions"].append(action)
                rec["new_chunk"].append(new_chunk)
                if explore_rng is not None:
                    rec["noisy"].append(noisy)
                rec["goals"].append(goal_predicates(env))
                rec["object_pos"].append(np.concatenate([obs[k] for k in pos_keys]))
            if done:
                break
            t += 1
        except Exception as e:  # main.py logs and ends the episode as a failure; we also record it
            logging.error(f"Caught exception: {e}")
            error = repr(e)
            break
    if args.dsrl and policy_steps > 0:  # final observation: the timeout bootstrap target; success = env done
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        state = np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
        client.infer({"observation/image": img, "observation/state": state, "dsrl_end": 1,
                      "success": int(bool(done) and not error)})
    raw = None
    if args.save_raw and policy_steps > 0:
        raw = {k: np.asarray(v) for k, v in rec.items()}
        # state and actions keep their own dtype (float64): float32 actions replay with ~1e-7 drift
        raw.update(sim_init=np.asarray(init_state), sim_start=sim_start, sim_end=env.get_sim_state(),
                   object_pos_keys=np.array(pos_keys), prompt=np.array(str(task_description)))
        if prefix is not None:
            raw.update(start_step=np.array(len(prefix["actions"])), prefix_actions=np.asarray(prefix["actions"]),
                       source_raw=np.array(prefix["source_raw"]), sim_wait=sim_wait)
        if switch_step is not None:
            raw.update(switch_step=np.array(switch_step))
        if w_idx is not None:
            raw.update(w_idx=np.array(int(w_idx)))
    return bool(done), policy_steps, error, frames, raw


def main(args: Args) -> None:
    np.random.seed(args.seed)
    out = pathlib.Path(args.out_dir)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        (out / "raw").mkdir(exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = args.max_steps if args.max_steps > 0 else MAX_STEPS[args.task_suite_name]
    task_ids = args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"matched noise: {args.matched_noise}")
    explore = args.explore_prob > 0
    if explore:
        assert not args.matched_noise, "exploration noise and matched noise together are not supported"
        assert 0 < args.explore_prob <= 1 and args.explore_scale > 0
        import json
        (out / "explore.json").write_text(json.dumps({"prob": args.explore_prob, "scale": args.explore_scale,
                                                      "seed": args.explore_seed}))
        logging.info(f"exploration: p {args.explore_prob} scale {args.explore_scale} seed {args.explore_seed}")

    csv_path = out / "episodes.csv"
    # a requeued SLURM job (preempted) resumes even when its stored batch script predates --args.resume
    resume = args.resume or os.environ.get("SLURM_RESTART_COUNT", "0") not in ("", "0")
    done_rows = load_done(csv_path) if resume and not args.fresh_layouts else {}
    if resume and args.fresh_layouts:
        logging.warning("--args.resume ignored with fresh layouts: starting over")
    if done_rows:
        logging.info(f"resume: {len(done_rows)} complete episodes kept from {csv_path}")
    starts = None
    if args.start_from:  # Changes 7
        assert not args.fresh_layouts and not args.matched_noise and not explore and not args.dsrl, \
            "--args.start-from runs alone (no fresh layouts / matched noise / exploration / dsrl)"
        assert len(task_ids) == 1, "--args.start-from: one task"
        with open(args.start_from, newline="") as fh:
            starts = list(csv.DictReader(fh))
        assert len({s["start_id"] for s in starts}) == len(starts), "start_id must be unique"
        logging.info(f"mid-task starts: {len(starts)} from {args.start_from}")
    hybrid = args.dsrl_base_prefix >= 0  # Changes 8
    if hybrid:
        assert args.dsrl and starts is None and 0 <= args.dsrl_base_prefix <= 1, \
            "--args.dsrl-base-prefix needs a DSRL eval server (--args.dsrl), a share in [0, 1], no start-from"
        logging.info(f"hybrid rollouts: base prefix U[0, {args.dsrl_base_prefix} x max_steps), then the DSRL actor")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS + (START_FIELDS if starts is not None else [])
                                + (PUSH_FIELDS if args.push_choices > 0 else []))
        writer.writeheader()
        for row in sorted(done_rows.values(), key=lambda r: int(r["episode"])):
            writer.writerow(row)
        fh.flush()
        episode = 0
        for task_id in task_ids:
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            ids = args.init_states if args.init_states is not None else list(range(len(init_states)))
            env_seed = args.seed
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, env_seed)
            n_ok = n = 0
            if starts is not None:
                plan = [(int(s["start_id"]), int(s["init_state"]), None) for s in starts]
            else:
                plan = ([(r, i, init_states[i]) for r in range(args.repeats) for i in ids] if not args.fresh_layouts
                        else [(r, -1, None) for r in range(args.fresh_layouts)])
            for p, (repeat, i, init) in enumerate(plan):
                prev = done_rows.get((task_id, i, repeat)) if (i >= 0 or starts is not None) else None
                if prev is not None:  # resumed: already complete in an earlier attempt of this job
                    n += 1
                    n_ok += int(prev["success"])
                    episode += 1
                    continue
                prefix, ep_max, extra = None, max_steps, {}
                if starts is not None:
                    s = starts[p]
                    k = int(s["start_step"])
                    with np.load(s["source_raw"]) as z:
                        prefix = {"actions": z["actions"][:k], "state_k": z["state"][k], "source_raw": s["source_raw"]}
                        init = z["sim_init"]
                        assert int(z["init_state_id"]) == i, (s, int(z["init_state_id"]))
                    ep_max = max_steps - k
                    if int(s["env_seed"]) != env_seed:  # the source's env seed (main.py: it can matter)
                        env.close()
                        env_seed = int(s["env_seed"])
                        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, env_seed)
                switch = None
                if hybrid:  # Changes 8: reproducible per episode, so a resumed run draws the same k
                    hrng = np.random.default_rng([args.dsrl_prefix_seed, task_id, i + 1, repeat, episode])
                    switch = int(hrng.uniform(0, args.dsrl_base_prefix * ep_max))
                w_idx = None
                if args.push_choices > 0:  # Changes 9: reproducible per episode, so a resumed run draws the same arm
                    w_idx = int(np.random.default_rng([args.push_seed, task_id, i + 1, repeat, episode])
                                .integers(args.push_choices))
                    extra = {**extra, "w_idx": w_idx}
                success, steps, error, frames, raw = run_episode(
                    env, client, init, task_description, args, ep_max,
                    noise_prefix=(task_id, i, repeat) if args.matched_noise else None,
                    explore_rng=(np.random.default_rng([args.explore_seed, task_id, i + 1, repeat, episode])
                                 if explore else None), prefix=prefix, switch_step=switch, w_idx=w_idx)
                if switch is not None:
                    logging.info(f"hybrid: base for the chunks before step {switch}, then the DSRL actor")
                if prefix is not None:
                    extra = {**extra, "start_step": k, "source_raw": s["source_raw"], "prefix_maxdiff": prefix.get("maxdiff", "")}
                    logging.info(f"start {repeat}: k {k}, hand-over state max |diff| {prefix.get('maxdiff')}")
                video = f"ep{episode:04d}_task{task_id}_init{'fresh' if i < 0 else f'{i:02d}'}_r{repeat}_{'success' if success else 'failure'}.mp4"
                imageio.mimwrite(out / "videos" / video, [np.asarray(x) for x in frames], fps=10)
                raw_name = ""
                if raw is not None:
                    raw_name = f"ep{episode:04d}.npz"
                    np.savez_compressed(out / "raw" / raw_name, init_state_id=i, **raw)
                goals_end = episode_goals_end(env, raw, steps)
                writer.writerow({"episode": episode, "suite": args.task_suite_name, "task_id": task_id,
                                 "task": task_description, "split": args.split_name, "init_state": i,
                                 "repeat": repeat, "env_seed": env_seed, "success": int(success),
                                 "timeout": int(not success and not error and steps == ep_max),
                                 "policy_steps": steps, "max_steps": ep_max, "goals_end": goals_end,
                                 "error": error, "video": video, "raw": raw_name, **extra})
                fh.flush()
                n += 1
                n_ok += success
                logging.info(f"ep {episode} task {task_id} init {i} r{repeat}: success={success} "
                             f"steps={steps} | task {n_ok}/{n}")
                episode += 1
            logging.info(f"Task {task_id} ({task_description}): {n_ok}/{n} = {n_ok / n:.3f}")
            env.close()
    logging.info(f"Done: {episode} episodes -> {csv_path}")


def load_done(csv_path):
    """{(task_id, init_state, repeat): row} of the complete rows of an earlier attempt's episodes.csv (a row cut off
    by preemption mid-write, or one missing a field, is dropped and its episode rerun)."""
    if not pathlib.Path(csv_path).exists():
        return {}
    done = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if None in row or any(row.get(k) is None for k in CSV_FIELDS) or row["success"] not in ("0", "1"):
                continue
            if row["raw"] and not row["raw"].endswith(".npz"):
                continue
            done[(int(row["task_id"]), int(row["init_state"]), int(row["repeat"]))] = row
    return done


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)  # main.py: seed affects object positions even with a fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from openpi examples/libero/main.py (itself from robosuite)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
