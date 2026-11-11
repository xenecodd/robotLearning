"""Watch a pretrained policy act in LIBERO live, inside the MuJoCo viewer.

The observation -> action path is the same as `lerobot-eval` (rollout() in lerobot_eval.py):
preprocess_observation -> env_preprocessor -> preprocessor -> select_action -> postprocessor
-> env_postprocessor -> env.step. Only the outer loop differs (keys, viewer, manual reset).

Keys (focus the viewer window). Function keys only: every letter, [ ] , Space, Backspace and Esc
already have a built-in MuJoCo viewer meaning (P = contact split, R = reflection, [ ] = camera...).
  F9   pause / resume            F10  single step (while paused)
  F11  restart same init state   F12  next init state
Close the window to quit.
At episode end (success or max_steps) it prints the result and freezes; F11 / F12 start the next one.
Do NOT use the viewer's own Backspace / "Reset" button: it puts every object at its XML default pose
(objects overlap and fly). If it happens anyway, the script detects it and does a proper LIBERO reset.
Mouse: double-click an object to select it, Ctrl + right-drag pushes it, Ctrl + left-drag rotates it.

Usage:
  bash interactive/run_live.sh --suite libero_object --task-id 0 --init-state 0
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

p = argparse.ArgumentParser()
p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
p.add_argument("--suite", default="libero_object")
p.add_argument("--task-id", type=int, default=0)
p.add_argument("--init-state", type=int, default=0, help="0..49, same as eval episode index")
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--n-action-steps", type=int, default=None, help="default: checkpoint value")
p.add_argument("--start-paused", action="store_true")
p.add_argument("--debug", action="store_true", help="every 10 steps: timing + mouse force + most moved object")
args = p.parse_args()

# --- env + policy, built exactly like lerobot_eval.eval_main ---
set_seed(args.seed)
env_cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task_id], init_states=True)
vec = make_env(env_cfg, n_envs=1, use_async_envs=False)[args.suite][args.task_id]
lib = vec.envs[0].unwrapped
max_steps = lib._max_episode_steps

policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
policy_cfg.pretrained_path = args.policy
policy_cfg.device = "cuda"
if args.n_action_steps is not None:
    policy_cfg.n_action_steps = args.n_action_steps
policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
policy.eval()
pre, post = make_pre_post_processors(
    policy_cfg=policy_cfg,
    pretrained_path=policy_cfg.pretrained_path,
    preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                            "rename_observations_processor": {"rename_map": {}}},
)
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
print(f"task: {lib.task_description} | max_steps {max_steps} | n_action_steps {policy.config.n_action_steps}")
print("keys: F9 pause/resume | F10 step | F11 restart | F12 next init state | close window = quit")

# --- keyboard state, written by the viewer thread, read by the main loop ---
KEY_F9, KEY_F10, KEY_F11, KEY_F12 = 298, 299, 300, 301  # GLFW key codes
state = {"paused": args.start_paused, "step_once": False, "cmd": None, "ended": False}


def on_key(keycode):
    if keycode == KEY_F11:
        state["cmd"] = "reset"
    elif keycode == KEY_F12:
        state["cmd"] = "next"
    elif state["ended"]:
        return  # pause/step mean nothing on a finished episode
    elif keycode == KEY_F9:
        state["paused"] = not state["paused"]
        print("paused" if state["paused"] else "resumed", flush=True)
    elif keycode == KEY_F10:
        state["step_once"] = True


def reset_episode(init_id):
    """Reset to a fixed init state. hard_reset rebuilds the MjModel, so return the new model/data."""
    lib.init_state_id = init_id
    obs, _ = vec.reset(seed=args.seed)
    policy.reset(); pre.reset(); post.reset()
    sim = lib._env.env.sim
    print(f"\n--- episode: task {args.task_id} init_state {init_id} seed {args.seed}", flush=True)
    return obs, sim.model._model, sim.data._data


def policy_action(obs):
    obs = preprocess_observation(obs)
    obs["task"] = list(vec.call("task_description"))
    obs = pre(env_pre(obs))
    with torch.inference_mode():
        action = post(policy.select_action(obs))
    return env_post({ACTION: action})[ACTION].to("cpu").numpy()


def free_bodies(m):
    return [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def viewer_was_reset(d, last_time):
    return d.time < last_time - 1e-9  # Backspace / Reset button sets d.time back to 0


init_id = args.init_state
obs, m, d = reset_episode(init_id)
viewer = mujoco.viewer.launch_passive(m, d, key_callback=on_key)
step, t0, last_time = 0, time.time(), d.time
t_pol = t_env = fmax = 0.0
p0 = {b: d.xpos[b].copy() for b in free_bodies(m)}
try:
    while viewer.is_running():
        if viewer_was_reset(d, last_time):
            print("viewer reset detected (Backspace / Reset button) -> doing a proper LIBERO reset", flush=True)
            state["cmd"] = "reset"
        cmd, state["cmd"] = state["cmd"], None
        if cmd:
            init_id = (init_id + (1 if cmd == "next" else 0)) % 50
            obs, m_new, d = reset_episode(init_id)
            if m_new is not m:  # new MjModel after hard reset -> reopen the viewer on it
                viewer.close()
                m = m_new
                viewer = mujoco.viewer.launch_passive(m, d, key_callback=on_key)
            step, t0, last_time, state["ended"] = 0, time.time(), d.time, False
            p0 = {b: d.xpos[b].copy() for b in free_bodies(m)}

        if state["ended"] or (state["paused"] and not state["step_once"]):
            viewer.sync()  # render + mouse, but no physics (a finished LIBERO env must not be stepped)
            fmax = max(fmax, np.abs(d.xfrc_applied).max())
            time.sleep(0.02)
            continue
        state["step_once"] = False

        ta = time.time()
        action = policy_action(obs)
        tb = time.time()
        obs, _, terminated, _, info = vec.step(action)
        fmax = max(fmax, np.abs(d.xfrc_applied).max())  # force the viewer applied during this step
        viewer.sync()
        t_pol += tb - ta
        t_env += time.time() - tb
        last_time = d.time
        step += 1
        success = bool(info["is_success"][0]) if "is_success" in info else False
        if state["paused"]:
            print(f"step {step}", flush=True)
        if args.debug and step % 10 == 0:
            moved = {b: np.linalg.norm(d.xpos[b] - p0[b]) * 100 for b in p0}
            b = max(moved, key=moved.get)
            print(f"[debug] step {step}: policy {t_pol / 10 * 1000:.0f} ms/step, env {t_env / 10 * 1000:.0f} ms/step | "
                  f"mouse force max {fmax:.2f} | most moved {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)} "
                  f"{moved[b]:.1f} cm", flush=True)
            t_pol = t_env = fmax = 0.0
        if terminated[0] or success or step >= max_steps:
            res = "SUCCESS" if success else "FAIL (timeout)" if step >= max_steps else "FAIL (done)"
            print(f"{res}  steps {step}/{max_steps}  wall {time.time() - t0:.1f}s "
                  f"({step / (time.time() - t0):.1f} steps/s)", flush=True)
            print("frozen on final state: F11 = same init state again, F12 = next", flush=True)
            state["ended"] = True
finally:
    viewer.close()
    vec.close()
