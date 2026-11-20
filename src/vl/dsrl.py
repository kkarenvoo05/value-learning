"""DSRL: noise-space SAC that steers a frozen pi0.5 (Wagenmaker, Nakamoto et al., CoRL 2025, arXiv 2506.15799).

Port of the official pi0 code (github.com/nakamotoo/dsrl_pi0 @ 7f48937: jaxrl2 PixelSACLearner + examples/
train_utils_sim.py) to run inside our openpi env (jax 0.5.3 / flax 0.10.2 / optax 0.2.4) with no new packages
[2026-11-10, Lars direction 1: calibration baseline]. Kept from the paper's LIBERO setup (Table 11 / run_libero.sh):
- action = one 32-d noise vector w in [-1, 1] (tanh-Gaussian, action_magnitude 1), repeated over the whole action
  horizon as pi0.5's flow-matching start noise ("single-step actor"); the first episodes use w ~ N(0, I) instead;
- pixel SAC: 64 x 64 agent-view image + 8-d proprio state; small CNN (32 x 4 filters, strides 2/1/1/1, VALID) ->
  Dense 50 + LayerNorm + tanh bottleneck (separate encoders for actor and critic); MLPs 128 x 3; 10 Q heads with
  LayerNorm, mean reduction (no clipped double Q); no entropy backup; target entropy -dim/2; temperature init 1;
  Adam actor 1e-4 / critic 3e-4 / temperature 3e-4; tau 0.005; batch 256; random-crop augmentation (pad 4, edge)
  on obs and next obs; sparse reward -1 per decision, 0 + terminal at success; discount 0.999 ** env steps per
  decision; UTD 20 gradient steps per decision, updates start after 500 transitions.
Changed [ours]: no colour jitter (the official learner applies it by default; sim renders are fixed); distrax
replaced by the closed-form tanh-Gaussian below; replay buffer is plain numpy; logging to JSON lines, not wandb;
one decision = one pi0.5 chunk of 5 executed steps (our replan rate; the paper's pi0 executes 20 of 50).
"""

from __future__ import annotations

import functools
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


@dataclass(frozen=True)  # hashable: static argument of the jitted update
class DSRLConfig:
    noise_dim: int = 32
    image_size: int = 64
    state_dim: int = 8
    hidden_dims: Sequence[int] = (128, 128, 128)
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    latent_dim: int = 50
    num_qs: int = 10
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    init_temperature: float = 1.0
    tau: float = 0.005
    action_magnitude: float = 1.0
    batch_size: int = 256
    crop_pad: int = 4


def orth(scale=float(np.sqrt(2))):
    return nn.initializers.orthogonal(scale)


class Encoder(nn.Module):
    features: Sequence[int]
    strides: Sequence[int]

    @nn.compact
    def __call__(self, pixels):  # (B, H, W, 3) uint8
        x = pixels.astype(jnp.float32) / 255.0
        for f, s in zip(self.features, self.strides):
            x = nn.relu(nn.Conv(f, (3, 3), strides=(s, s), padding="VALID", kernel_init=orth())(x))
        return x.reshape((x.shape[0], -1))


class Trunk(nn.Module):
    """PixelMultiplexer: CNN -> Dense + LayerNorm + tanh bottleneck, concatenated with the proprio state."""
    cfg: DSRLConfig

    @nn.compact
    def __call__(self, pixels, state):
        x = Encoder(self.cfg.cnn_features, self.cfg.cnn_strides)(pixels)
        x = nn.tanh(nn.LayerNorm()(nn.Dense(self.cfg.latent_dim, kernel_init=nn.initializers.xavier_normal())(x)))
        return jnp.concatenate([x, state], -1)  # jaxrl2 flattens dict keys sorted: pixels, then state


class Actor(nn.Module):
    cfg: DSRLConfig

    @nn.compact
    def __call__(self, pixels, state):
        x = Trunk(self.cfg)(pixels, state)
        for h in self.cfg.hidden_dims:
            x = nn.relu(nn.Dense(h, kernel_init=orth(1.0))(x))
        mean = nn.Dense(self.cfg.noise_dim, kernel_init=orth(1e-2))(x)
        log_std = jnp.clip(nn.Dense(self.cfg.noise_dim, kernel_init=orth(1e-2))(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class QHead(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, feat, action):
        x = jnp.concatenate([action, feat], -1)  # jaxrl2 sorted keys: 'actions' before 'states'
        for h in self.hidden_dims:
            x = nn.relu(nn.LayerNorm()(nn.Dense(h, kernel_init=orth(1.0))(x)))
        return nn.Dense(1, kernel_init=orth(1.0))(x)[..., 0]


class Critic(nn.Module):
    cfg: DSRLConfig

    @nn.compact
    def __call__(self, pixels, state, action):
        feat = Trunk(self.cfg)(pixels, state)
        heads = nn.vmap(QHead, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0,
                        axis_size=self.cfg.num_qs)
        return heads(self.cfg.hidden_dims)(feat, action)  # (num_qs, B)


class Temperature(nn.Module):
    initial: float = 1.0

    @nn.compact
    def __call__(self):
        return jnp.exp(self.param("log_temp", lambda _: jnp.full((), jnp.log(self.initial))))


def tanh_sample(key, mean, log_std, magnitude):
    """Sample a = magnitude * tanh(u), u ~ N(mean, std); log-prob with the tanh and scale Jacobians (as distrax)."""
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(key, mean.shape)
    logp_u = -0.5 * (((u - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    log_det = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))  # log(1 - tanh(u)^2), stable
    logp = (logp_u - log_det).sum(-1) - mean.shape[-1] * jnp.log(magnitude)
    return magnitude * jnp.tanh(u), logp


def random_crop(key, imgs, pad):
    """DrQ-style random shift, edge padding (jaxrl2 batched_random_crop)."""
    def one(k, img):
        xy = jax.random.randint(k, (2,), 0, 2 * pad + 1)
        p = jnp.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        return jax.lax.dynamic_slice(p, (xy[0], xy[1], 0), img.shape)
    return jax.vmap(one)(jax.random.split(key, imgs.shape[0]), imgs)


@functools.partial(jax.jit, static_argnames=("cfg",))
def _update(rng, actor, critic, target_params, temp, batch, cfg: DSRLConfig, target_entropy):
    rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
    pix = random_crop(k1, batch["pixels"], cfg.crop_pad)
    npix = random_crop(k2, batch["next_pixels"], cfg.crop_pad)
    s, ns = batch["state"], batch["next_state"]
    alpha = temp.apply_fn({"params": temp.params})

    nmean, nlog = actor.apply_fn({"params": actor.params}, npix, ns)
    na, _ = tanh_sample(k3, nmean, nlog, cfg.action_magnitude)
    next_q = critic.apply_fn({"params": target_params}, npix, ns, na).mean(0)
    target_q = batch["reward"] + batch["discount"] * batch["mask"] * next_q

    def critic_loss(p):
        qs = critic.apply_fn({"params": p}, pix, s, batch["action"])
        return ((qs - target_q) ** 2).mean(), qs.mean()
    (closs, qmean), g = jax.value_and_grad(critic_loss, has_aux=True)(critic.params)
    critic = critic.apply_gradients(grads=g)
    target_params = optax.incremental_update(critic.params, target_params, cfg.tau)

    def actor_loss(p):
        mean, log_std = actor.apply_fn({"params": p}, pix, s)
        a, logp = tanh_sample(k4, mean, log_std, cfg.action_magnitude)
        q = critic.apply_fn({"params": critic.params}, pix, s, a).mean(0)
        return (logp * alpha - q).mean(), (-logp.mean(), jnp.exp(log_std).mean())
    (aloss, (entropy, std)), g = jax.value_and_grad(actor_loss, has_aux=True)(actor.params)
    actor = actor.apply_gradients(grads=g)

    def temp_loss(p):
        return temp.apply_fn({"params": p}) * (entropy - target_entropy)
    tloss, g = jax.value_and_grad(temp_loss)(temp.params)
    temp = temp.apply_gradients(grads=g)
    info = dict(critic_loss=closs, q=qmean, target_q=target_q.mean(), actor_loss=aloss, entropy=entropy,
                actor_std=std, temperature=alpha, temp_loss=tloss)
    return rng, actor, critic, target_params, temp, info


@functools.partial(jax.jit, static_argnames=("apply_fn", "magnitude"))
def _sample(apply_fn, params, key, pixels, state, magnitude):
    mean, log_std = apply_fn({"params": params}, pixels, state)
    return tanh_sample(key, mean, log_std, magnitude)[0]


@functools.partial(jax.jit, static_argnames=("apply_fn",))
def _q(apply_fn, params, pixels, state, action):
    return apply_fn({"params": params}, pixels, state, action)


class DSRLAgent:
    def __init__(self, seed: int = 0, cfg: DSRLConfig | None = None):
        cfg = cfg or DSRLConfig()
        self.cfg = cfg = replace(cfg, hidden_dims=tuple(cfg.hidden_dims), cnn_features=tuple(cfg.cnn_features),
                                 cnn_strides=tuple(cfg.cnn_strides))  # tuples: hashable for jit (pickle gives lists)
        rng = jax.random.PRNGKey(seed)
        rng, ka, kc, kt = jax.random.split(rng, 4)
        pix = jnp.zeros((1, cfg.image_size, cfg.image_size, 3), jnp.uint8)
        st = jnp.zeros((1, cfg.state_dim), jnp.float32)
        act = jnp.zeros((1, cfg.noise_dim), jnp.float32)
        actor_def, critic_def, temp_def = Actor(cfg), Critic(cfg), Temperature(cfg.init_temperature)
        self.actor = TrainState.create(apply_fn=actor_def.apply, params=actor_def.init(ka, pix, st)["params"],
                                       tx=optax.adam(cfg.actor_lr))
        cp = critic_def.init(kc, pix, st, act)["params"]
        self.critic = TrainState.create(apply_fn=critic_def.apply, params=cp, tx=optax.adam(cfg.critic_lr))
        self.target_params = jax.tree.map(jnp.array, cp)
        self.temp = TrainState.create(apply_fn=temp_def.apply, params=temp_def.init(kt)["params"],
                                      tx=optax.adam(cfg.temp_lr))
        self.target_entropy = -cfg.noise_dim / 2
        self.rng = rng

    def sample(self, pixels: np.ndarray, state: np.ndarray) -> np.ndarray:
        """One noise vector (noise_dim,) for one observation (H, W, 3) uint8 + (state_dim,)."""
        self.rng, k = jax.random.split(self.rng)
        a = _sample(self.actor.apply_fn, self.actor.params, k, jnp.asarray(pixels)[None],
                    jnp.asarray(state, jnp.float32)[None], self.cfg.action_magnitude)
        return np.asarray(a[0], np.float32)

    def q(self, pixels: np.ndarray, state: np.ndarray, w: np.ndarray) -> np.ndarray:
        """[2026-11-13, DSRL G1] Q(s, w) of every head, (num_qs, B), for B observations (B, H, W, 3) uint8 + (B,
        state_dim) and noise vectors (B, noise_dim). No random crop (that is train-time augmentation only); the
        SAC losses use the mean over heads."""
        return np.asarray(_q(self.critic.apply_fn, self.critic.params, jnp.asarray(pixels),
                             jnp.asarray(state, jnp.float32), jnp.asarray(w, jnp.float32)), np.float32)

    def update(self, batch: dict) -> dict:
        self.rng, self.actor, self.critic, self.target_params, self.temp, info = _update(
            self.rng, self.actor, self.critic, self.target_params, self.temp, batch, self.cfg, self.target_entropy)
        return info

    def save(self, path):
        blob = {"cfg": asdict(self.cfg), "actor": self.actor.params, "critic": self.critic.params,
                "target": self.target_params, "temp": self.temp.params, "rng": self.rng}
        Path(path).write_bytes(pickle.dumps(jax.device_get(blob)))

    def full_state(self) -> dict:
        """[2026-11-12] Everything update() and sample() depend on (weights, Adam moments, steps, target, rng), for a
        resumable run; load(path) keeps restoring weights only."""
        ts = {name: {"step": t.step, "params": t.params, "opt_state": t.opt_state}
              for name, t in (("actor", self.actor), ("critic", self.critic), ("temp", self.temp))}
        return jax.device_get({"cfg": asdict(self.cfg), **ts, "target": self.target_params, "rng": self.rng})

    @classmethod
    def from_full_state(cls, blob: dict) -> "DSRLAgent":
        agent = cls(0, DSRLConfig(**blob["cfg"]))
        for name in ("actor", "critic", "temp"):
            t = getattr(agent, name)
            setattr(agent, name, t.replace(step=blob[name]["step"], params=blob[name]["params"],
                                           opt_state=blob[name]["opt_state"]))
        agent.target_params, agent.rng = blob["target"], jnp.asarray(blob["rng"])
        return agent

    @classmethod
    def load(cls, path, seed: int = 0) -> "DSRLAgent":
        """Weights only (optimizer moments restart): enough to serve the actor in eval mode."""
        blob = pickle.loads(Path(path).read_bytes())
        agent = cls(seed, DSRLConfig(**blob["cfg"]))
        agent.actor = agent.actor.replace(params=blob["actor"])
        agent.critic = agent.critic.replace(params=blob["critic"])
        agent.target_params, agent.temp = blob["target"], agent.temp.replace(params=blob["temp"])
        return agent


class ReplayBuffer:
    """Transitions (s, w, r, s', mask, discount); next obs stored explicitly (simple; ~25 KB per transition)."""

    def __init__(self, capacity: int, cfg: DSRLConfig, seed: int = 0):
        H = cfg.image_size
        self.d = {"pixels": np.zeros((capacity, H, H, 3), np.uint8),
                  "next_pixels": np.zeros((capacity, H, H, 3), np.uint8),
                  "state": np.zeros((capacity, cfg.state_dim), np.float32),
                  "next_state": np.zeros((capacity, cfg.state_dim), np.float32),
                  "action": np.zeros((capacity, cfg.noise_dim), np.float32),
                  **{k: np.zeros(capacity, np.float32) for k in ("reward", "mask", "discount")}}
        self.capacity, self.n, self.i = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def add_episode(self, pixels, states, actions, success: bool, discount: float, final_pixels, final_state):
        """One episode of T decisions (DSRL collect_traj + add_online_data_to_buffer): reward -1 per decision,
        the last decision of a success gets 0 and mask 0; a timeout bootstraps from the final observation."""
        T = len(actions)
        assert T == len(pixels) == len(states) and T > 0
        rew = -np.ones(T, np.float32)
        mask = np.ones(T, np.float32)
        if success:
            rew[-1], mask[-1] = 0.0, 0.0
        nxt_p = list(pixels[1:]) + [final_pixels]
        nxt_s = list(states[1:]) + [final_state]
        for t in range(T):
            j = self.i
            for k, v in (("pixels", pixels[t]), ("next_pixels", nxt_p[t]), ("state", states[t]),
                         ("next_state", nxt_s[t]), ("action", actions[t]), ("reward", rew[t]), ("mask", mask[t]),
                         ("discount", discount)):
                self.d[k][j] = v
            self.i = (self.i + 1) % self.capacity
            self.n = min(self.n + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.integers(0, self.n, batch_size)
        return {k: v[idx] for k, v in self.d.items()}

    def save_chunks(self, directory, chunks: list[str]) -> dict:
        """[2026-11-12] Append-only save for a resumable run: writes rows [end of the last chunk, n) as one new .npz
        chunk (~25 KB per transition, so ~100 MB per 50 episodes; the whole buffer is several GB, and the client's
        websocket gives up if a request blocks for ~40 s). Returns the meta to store with the checkpoint (chunk list,
        n, i, rng state). Needs a buffer that never wrapped (capacity 250k >= 1800 episodes x 104 decisions)."""
        assert self.i == self.n < self.capacity, "buffer wrapped: chunked saving needs a larger capacity"
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        start = int(chunks[-1].split("_")[2].split(".")[0]) if chunks else 0
        chunks = list(chunks)
        if self.n > start:
            name = f"rows_{start:07d}_{self.n:07d}.npz"
            tmp = directory / (name + ".tmp.npz")
            np.savez(tmp, **{k: v[start: self.n] for k, v in self.d.items()})
            tmp.replace(directory / name)
            chunks.append(name)
        return {"chunks": chunks, "n": self.n, "i": self.i, "capacity": self.capacity,
                "rng": self.rng.bit_generator.state}

    def load_chunks(self, directory, meta: dict) -> None:
        directory = Path(directory)
        assert meta["capacity"] == self.capacity, (meta["capacity"], self.capacity)
        pos = 0
        for name in meta["chunks"]:
            a, b = (int(x) for x in name[len("rows_"):-len(".npz")].split("_"))
            assert a == pos, (name, pos)
            with np.load(directory / name) as z:
                for k in self.d:
                    assert len(z[k]) == b - a, (name, k)
                    self.d[k][a:b] = z[k]
            pos = b
        assert pos == meta["n"], (pos, meta["n"])
        self.n, self.i = meta["n"], meta["i"]
        self.rng.bit_generator.state = meta["rng"]
