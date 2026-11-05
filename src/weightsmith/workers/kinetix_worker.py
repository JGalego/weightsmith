"""Out-of-process Kinetix evaluation worker for Weightsmith's ``rollouts`` suites.

Runs a fixed Kinetix actor (the symbolic-flat MLP of ``kinetix/models/actor_critic.py``) in one
Kinetix level and reports how the episodes end. It runs in its own environment (see
``README.md`` and ``kinetix_requirements.txt``), not in Weightsmith's::

    ~/.cache/weightsmith/workers/kinetix/venv/bin/python -I kinetix_worker.py < request.json

Request, one JSON object on stdin::

    {"weights": "/abs/path/model.safetensors", "level": "l/h2_car_ramp", "episodes": 1024,
     "seed": 1, "mode": "sample" | "argmax"}

The safetensors file holds (out, in) tensors: ``embed.weight`` (128, 679), ``embed.bias``,
``actor0.weight`` .. ``actor4.weight`` (128, 128), their biases, ``actor_head.weight`` (16, 128)
and ``actor_head.bias``.

Response, one JSON object on stdout (diagnostics go to stderr)::

    {"episodes": N, "success": k, "red": k, "timeout": k, "invalid": k, "mean_return": x,
     "mean_length": x, "env_steps": k, "seconds": x, ...}

or ``{"error": "..."}`` with exit code 2 for a bad request and 1 for any other failure.

The environment is built the way Kinetix's ``experiments/ppo.py`` builds it for the hydra
overrides ``env=symbolic env_size=l train_levels.train_levels_list=[<level>]``:
``generate_params_from_config`` with the ``env/symbolic.yaml`` and ``env_size/l.yaml`` values,
``make_reset_fn_from_config`` in "list" mode, and ``make_kinetix_env(MULTI_DISCRETE,
SYMBOLIC_FLAT, ...)``. Auto-reset is off: every episode runs from the level's start state until
its first terminal step (or the step limit), and later steps are masked out.

Episode i is seeded with ``fold_in(PRNGKey(seed), i)`` and step t with ``fold_in(episode_key,
t)``, so a request gives the same result every time, and the first k episodes of a larger request
are the k episodes of a smaller one.
"""

from __future__ import annotations

import json
import os
import sys
import time

# CPU only, one XLA host device per core so the episode batch can be split across cores with pmap.
# Measured on 12 cores (252 episodes x 256 steps): jaxlib 0.4.38's default XLA:CPU "thunk" runtime
# takes ~125 s, the legacy runtime ~53 s, and the legacy runtime with single-threaded Eigen per
# device (the devices already use every core) ~30 s. Both are selected unless the caller's XLA_FLAGS
# say otherwise.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _xla_flags:
    _n_dev = int(os.environ.get("WEIGHTSMITH_KINETIX_DEVICES", os.cpu_count() or 1))
    _xla_flags += f" --xla_force_host_platform_device_count={_n_dev}"
if "xla_cpu_use_thunk_runtime" not in _xla_flags:
    _xla_flags += " --xla_cpu_use_thunk_runtime=false"
if "xla_cpu_multi_thread_eigen" not in _xla_flags:
    _xla_flags += " --xla_cpu_multi_thread_eigen=false"
os.environ["XLA_FLAGS"] = _xla_flags.strip()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ENV_SIZE = "l"  # hydra `env_size=l`
DENSE_REWARD_SCALE = 0.2  # configs/env/symbolic.yaml
OBS_DIM = 679
N_LOGITS = 16
HIDDEN = 128
ACTOR_DEPTH = 5
LAYERS = ["embed"] + [f"actor{k}" for k in range(ACTOR_DEPTH)] + ["actor_head"]
SHAPES = {"embed": (HIDDEN, OBS_DIM), "actor_head": (N_LOGITS, HIDDEN)}
SHAPES.update({f"actor{k}": (HIDDEN, HIDDEN) for k in range(ACTOR_DEPTH)})

# Outcome codes.
RUNNING, SUCCESS, RED, TIMEOUT, INVALID = 0, 1, 2, 3, 4

CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "weightsmith", "workers", "kinetix", "jax_cache"
)


def log(msg: str) -> None:
    print(f"[kinetix_worker] {msg}", file=sys.stderr, flush=True)


class RequestError(ValueError):
    pass


def parse_request(raw: str) -> dict:
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RequestError(f"request is not JSON: {e}") from None
    if not isinstance(req, dict):
        raise RequestError("request must be a JSON object")
    weights = req.get("weights")
    if not isinstance(weights, str) or not os.path.isabs(weights) or not os.path.isfile(weights):
        raise RequestError(
            f"'weights' must be an absolute path to an existing file, got {weights!r}"
        )
    level = req.get("level", "l/h2_car_ramp")
    if not isinstance(level, str) or not level:
        raise RequestError("'level' must be a non-empty string")
    episodes = req.get("episodes", 1024)
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
        raise RequestError("'episodes' must be a positive integer")
    seed = req.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise RequestError("'seed' must be an integer in [0, 2**32)")
    mode = req.get("mode", "sample")
    if mode not in ("sample", "argmax"):
        raise RequestError("'mode' must be 'sample' or 'argmax'")
    return {"weights": weights, "level": level, "episodes": episodes, "seed": seed, "mode": mode}


def load_actor(path: str):
    """Read the (out, in) safetensors actor; return float32 (in, out) kernels and biases, in
    layer order."""
    import numpy as np
    from safetensors.numpy import load_file

    tensors = load_file(path)
    params = []
    for name in LAYERS:
        w, b = tensors.get(f"{name}.weight"), tensors.get(f"{name}.bias")
        if w is None or b is None:
            raise RequestError(f"weights file lacks {name}.weight / {name}.bias")
        if tuple(w.shape) != SHAPES[name] or tuple(b.shape) != (SHAPES[name][0],):
            raise RequestError(
                f"{name}: expected weight {SHAPES[name]} and bias ({SHAPES[name][0]},), "
                f"got {w.shape} and {b.shape}"
            )
        w = np.ascontiguousarray(np.asarray(w, dtype=np.float32).T)
        b = np.asarray(b, dtype=np.float32)
        if not (np.isfinite(w).all() and np.isfinite(b).all()):
            raise RequestError(f"{name} has non-finite entries")
        params.append((w, b))
    return params


def build_env(level: str):
    """The env, env params and level start state, as Kinetix's PPO builds them for
    env=symbolic env_size=l."""
    import jax
    from kinetix.environment import make_kinetix_env, make_reset_fn_from_config
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.environment.utils import static_env_params_from_size
    from kinetix.util.config import generate_params_from_config

    # configs/env/symbolic.yaml + configs/env_size/l.yaml (env_size_type: predefined), merged the
    # way kinetix/util/config.py:normalise_config merges them (static_env_params_from_size(...,
    # return_dict=True)).
    _, size_cfg = static_env_params_from_size(ENV_SIZE, return_dict=True)
    config = {
        "action_type": ActionType.MULTI_DISCRETE,
        "observation_type": ObservationType.SYMBOLIC_FLAT,
        "dense_reward_scale": DENSE_REWARD_SCALE,
        "dummy_env": False,
        "env_size_type": "predefined",
        "train_level_mode": "list",
        "train_levels_list": [level],
        **size_cfg,
    }
    env_params, static_env_params = generate_params_from_config(config)
    reset_fn = make_reset_fn_from_config(config, env_params, static_env_params)
    env = make_kinetix_env(
        config["action_type"],
        config["observation_type"],
        reset_fn,
        env_params,
        static_env_params,
        auto_reset=False,
        create_dummy_env=False,
    )
    obs0, state0 = env.reset(jax.random.PRNGKey(0), env_params)
    return env, env_params, static_env_params, obs0, state0


def make_runner(env, env_params, static_env_params, max_steps: int, mode: str):
    """A function (params, ep_keys[B], obs0, state0) -> per-episode (outcome, return, length),
    batched over B."""
    import jax
    import jax.numpy as jnp

    n_motor = static_env_params.num_motor_bindings
    n_thrust = static_env_params.num_thruster_bindings
    sizes = [3] * n_motor + [2] * n_thrust
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    assert sum(sizes) == N_LOGITS, sizes
    scale = env_params.dense_reward_scale

    def logits_fn(params, x):
        (w, b), hidden, (wh, bh) = params[0], params[1:-1], params[-1]
        h = jax.nn.relu(x @ w + b)
        for wk, bk in hidden:
            h = jnp.tanh(h @ wk + bk)
        return h @ wh + bh

    def choose(key, logits):
        # Mirrors kinetix/models/action_spaces.py:MultiDiscreteActionDistribution: one key per
        # entry, each entry drawn independently from the softmax of its own slice of the logits.
        keys = jax.random.split(key, len(sizes))
        out = []
        for j, (s, n) in enumerate(zip(starts, sizes, strict=True)):
            lj = logits[s : s + n]
            if mode == "sample":
                out.append(jax.random.categorical(keys[j], lj))
            else:
                out.append(jnp.argmax(lj))
        return jnp.stack(out).astype(jnp.int32)

    def has_nan(state):
        leaves = [
            x for x in jax.tree.leaves(state) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
        ]
        return jnp.any(jnp.stack([jnp.isnan(x).any() for x in leaves]))

    def one_step(params, ep_key, t, obs, state):
        logits = logits_fn(params, obs)
        action = choose(jax.random.fold_in(ep_key, t), logits)
        obs2, state2, reward, done, info = env.step(
            jax.random.fold_in(ep_key, t), state, action, env_params
        )
        # Split the step reward into its terminal (+1 / -1 / 0) and dense parts, the way
        # kinetix/environment/env.py:KinetixEnv.engine_step adds them, to tell a red contact from
        # the rest.
        prev = state.last_distance
        dense = jnp.nan_to_num(-(info["distance"] - prev) * scale, nan=0.0, posinf=0.0, neginf=0.0)
        dense = jnp.where((prev == -1.0) | (scale == 0.0), 0.0, dense)
        terminal = reward - dense
        success = info["GoalR"]
        red = (~success) & (terminal < -0.5)
        invalid = (~success) & (~red) & has_nan(state2)
        code = jnp.where(
            success, SUCCESS, jnp.where(red, RED, jnp.where(invalid, INVALID, TIMEOUT))
        )
        return obs2, state2, reward, done, code.astype(jnp.int32)

    batched_step = jax.vmap(one_step, in_axes=(None, 0, None, 0, 0))

    def run(params, ep_keys, obs0, state0):
        b = ep_keys.shape[0]
        obs = jnp.broadcast_to(obs0, (b, *obs0.shape))
        state = jax.tree.map(lambda x: jnp.broadcast_to(x, (b, *jnp.shape(x))), state0)
        carry = (
            jnp.int32(0),
            obs,
            state,
            jnp.zeros(b, jnp.int32),  # outcome
            jnp.zeros(b, jnp.float32),  # return
            jnp.zeros(b, jnp.int32),  # length
        )

        def cond(c):
            t, _, _, outcome, _, _ = c
            return (t < max_steps) & jnp.any(outcome == RUNNING)

        def body(c):
            t, obs, state, outcome, ret, length = c
            obs2, state2, reward, done, code = batched_step(params, ep_keys, t, obs, state)
            live = outcome == RUNNING
            ret = ret + jnp.where(live, reward, 0.0)
            length = length + live.astype(jnp.int32)
            # An episode still running at the step limit is a timeout even if Kinetix's own limit
            # differs.
            ended = done | (t + 1 >= max_steps)
            code = jnp.where(done, code, TIMEOUT)
            outcome = jnp.where(live & ended, code, outcome)

            def keep(new, old):
                m = live.reshape((b,) + (1,) * (new.ndim - 1))
                return jnp.where(m, new, old)

            obs = keep(obs2, obs)
            state = jax.tree.map(keep, state2, state)
            return (t + 1, obs, state, outcome, ret, length)

        _, _, _, outcome, ret, length = jax.lax.while_loop(cond, body, carry)
        return outcome, ret, length

    return run


def evaluate(req: dict) -> dict:
    t_start = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception as e:  # the cache is an optimisation only
        log(f"compilation cache disabled: {e}")

    params_np = load_actor(req["weights"])
    env, env_params, static_env_params, obs0, state0 = build_env(req["level"])
    if obs0.shape != (OBS_DIM,):
        raise RuntimeError(f"observation has shape {obs0.shape}, expected ({OBS_DIM},)")
    max_steps = int(env_params.max_timesteps)

    devices = jax.local_devices()
    n_dev = max(1, min(len(devices), req["episodes"]))
    per_dev = -(-req["episodes"] // n_dev)
    total = per_dev * n_dev

    base = jax.random.PRNGKey(req["seed"])
    ep_keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.arange(total, dtype=jnp.uint32))
    ep_keys = ep_keys.reshape((n_dev, per_dev, *ep_keys.shape[1:]))
    params = jax.tree.map(jnp.asarray, [tuple(p) for p in params_np])

    run = make_runner(env, env_params, static_env_params, max_steps, req["mode"])
    prun = jax.pmap(run, in_axes=(None, 0, None, None), devices=devices[:n_dev])
    t_setup = time.perf_counter()
    compiled = prun.lower(params, ep_keys, obs0, state0).compile()
    t_compiled = time.perf_counter()
    outcome, ret, length = compiled(params, ep_keys, obs0, state0)
    outcome = np.asarray(outcome).reshape(-1)[: req["episodes"]]
    ret = np.asarray(ret, dtype=np.float64).reshape(-1)[: req["episodes"]]
    length = np.asarray(length).reshape(-1)[: req["episodes"]]
    t_end = time.perf_counter()

    if (outcome == RUNNING).any():
        raise RuntimeError("some episodes did not finish within the step limit")
    n = int(req["episodes"])
    result = {
        "episodes": n,
        "success": int((outcome == SUCCESS).sum()),
        "red": int((outcome == RED).sum()),
        "timeout": int((outcome == TIMEOUT).sum()),
        "invalid": int((outcome == INVALID).sum()),
        "mean_return": float(ret.mean()),
        "mean_length": float(length.mean()),
        "env_steps": int(length.sum()),
        "level": req["level"],
        "mode": req["mode"],
        "seed": req["seed"],
        "max_steps": max_steps,
        "devices": n_dev,
        "seconds": round(t_end - t_start, 3),
        "setup_seconds": round(t_setup - t_start, 3),
        "compile_seconds": round(t_compiled - t_setup, 3),
        "rollout_seconds": round(t_end - t_compiled, 3),
    }
    log(
        f"{n} episodes ({req['mode']}, seed {req['seed']}) on {n_dev} CPU devices: "
        f"success {result['success']}, red {result['red']}, timeout {result['timeout']}, "
        f"invalid {result['invalid']}; {result['seconds']} s"
    )
    return result


def main() -> int:
    # Kinetix and its dependencies may print to stdout (Python or native code). Point fd 1 at stderr
    # while working and write the one JSON result to the original stdout at the end.
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    code = 0
    try:
        result = evaluate(parse_request(sys.stdin.read()))
    except RequestError as e:
        result, code = {"error": str(e)}, 2
    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        result, code = {"error": f"{type(e).__name__}: {e}"}, 1
    out.write(json.dumps(result) + "\n")
    out.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())
