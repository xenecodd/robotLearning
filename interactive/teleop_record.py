"""Drive the LIBERO robot with the numpad inside the MuJoCo viewer and record demos as a LeRobotDataset.

The recorded dataset uses the same schema as the local `lerobot/libero` subset (data/libero_task0):
observation.images.image (256x256x3, rotated 180 deg like LiberoProcessorStep does), observation.state
(eef pos 3 + axis-angle 3 + gripper qpos 2), action (7D OSC delta in [-1, 1]), task. One env step = one frame
(the source dataset does the same; its fps=10 is only a label).

The viewer only reports key presses, never releases, so motion is "sticky": each press adds one speed level
on that axis and the robot keeps moving until you press the opposite key or Num 5.
Keys (focus the viewer window, NumLock state does not matter):
  Num 8 / 2  +x / -x        Num 4 / 6  +y / -y        Num 9 / 3  up / down
  Num 7 / 1  yaw + / -      Num 5      stop all motion  Num 0      gripper open / close
  F9  pause / resume        F11  discard, restart same init state      F12  next init state
Physics does not run until the first motion or gripper key, so there are no idle frames at the start.
A successful episode is saved immediately (unless --no-record); failures (timeout) are discarded.
Do NOT use Backspace / the viewer's "Reset" button (objects jump to XML defaults); it is detected and
turned into a proper restart.

Usage:
  bash interactive/run_teleop.sh --task-id 0 --init-state 0 --no-record   # practice, nothing saved
  bash interactive/run_teleop.sh --task-id 0 --init-state 0               # record into data/teleop_demos
"""
import argparse
import json
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_object")
p.add_argument("--task-id", type=int, default=0)
p.add_argument("--init-state", type=int, default=0, help="0..49")
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--root", default="data/teleop_demos", help="output dataset dir (appended to if it exists)")
p.add_argument("--schema-from", default="data/libero_task0", help="dataset whose features are copied")
p.add_argument("--no-record", action="store_true", help="practice mode, nothing is written")
p.add_argument("--speed", type=float, default=0.25, help="action added per key press (translation)")
p.add_argument("--rot-speed", type=float, default=0.1, help="action added per key press (yaw)")
p.add_argument("--max-steps", type=int, default=1000,
               help="episode limit; the eval limit (280 for libero_object) is only 14 s of real time")
p.add_argument("--debug", action="store_true", help="print the GLFW code of unmapped keys")
args = p.parse_args()

# --- env: render the scene camera at the dataset resolution ---
env_cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task_id], init_states=True,
                          camera_name="agentview_image", observation_height=256, observation_width=256)
vec = make_env(env_cfg, n_envs=1, use_async_envs=False)[args.suite][args.task_id]
lib = vec.envs[0].unwrapped  # single LiberoEnv, unbatched obs
max_steps = args.max_steps
task = lib.task_description

# --- dataset with the same features as the source dataset ---
ds = None
if not args.no_record:
    root = Path(args.root)
    if (root / "meta" / "info.json").exists():
        ds = LeRobotDataset.resume(repo_id="local/teleop_demos", root=root)
    else:
        src = json.load(open(Path(args.schema_from) / "meta" / "info.json"))
        keep = ("observation.images.image", "observation.state", "action")
        features = {k: {f: src["features"][k][f] for f in ("dtype", "shape", "names")} for k in keep}
        ds = LeRobotDataset.create(repo_id="local/teleop_demos", fps=int(src["fps"]), features=features,
                                   root=root, robot_type="panda", use_videos=True)
    print(f"recording into {args.root} ({ds.meta.total_episodes} episodes so far)")

# --- keyboard: GLFW key codes ---
KP = {i: 320 + i for i in range(10)}  # KP_0..KP_9
KEY_F9, KEY_F11, KEY_F12 = 298, 300, 301
# key -> (action index, sign). Index: 0 x, 1 y, 2 z, 5 yaw (OSC delta in the robot base frame)
MOTION = {KP[8]: (0, +1), KP[2]: (0, -1), KP[4]: (1, +1), KP[6]: (1, -1),
          KP[9]: (2, +1), KP[3]: (2, -1), KP[7]: (5, +1), KP[1]: (5, -1)}
state = {"vel": np.zeros(6), "grip": -1.0, "paused": False, "started": False, "ended": False, "cmd": None}


def on_key(keycode):
    if keycode == KEY_F11:
        state["cmd"] = "reset"
    elif keycode == KEY_F12:
        state["cmd"] = "next"
    elif state["ended"]:
        return
    elif keycode == KEY_F9:
        state["paused"] = not state["paused"]
        print("paused" if state["paused"] else "resumed", flush=True)
    elif keycode in MOTION:
        i, s = MOTION[keycode]
        step = args.rot_speed if i == 5 else args.speed
        state["vel"][i] = np.clip(state["vel"][i] + s * step, -1.0, 1.0)
        state["started"] = True
        print(f"vel xyz {state['vel'][:3].round(2)} yaw {state['vel'][5]:.2f}", flush=True)
    elif keycode == KP[5]:
        state["vel"][:] = 0.0
        print("stop", flush=True)
    elif keycode == KP[0]:
        state["grip"] = -state["grip"]  # LIBERO: -1 open, +1 close
        state["started"] = True
        print("gripper", "close" if state["grip"] > 0 else "open", flush=True)
    elif args.debug:
        print(f"unmapped key {keycode}", flush=True)


def quat2axisangle(q):
    """(x, y, z, w) quaternion -> axis-angle, same math as LiberoProcessorStep / robosuite."""
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


def to_frame(obs, action):
    rs = obs["robot_state"]
    st = np.concatenate([rs["eef"]["pos"], quat2axisangle(rs["eef"]["quat"]), rs["gripper"]["qpos"]])
    return {"observation.images.image": np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1]),
            "observation.state": st.astype(np.float32), "action": action.astype(np.float32), "task": task}


def reset_episode(init_id):
    lib.init_state_id = init_id
    obs, _ = lib.reset(seed=args.seed)
    state.update(vel=np.zeros(6), grip=-1.0, started=False, ended=False, paused=False)
    if ds is not None and ds.has_pending_frames():
        ds.clear_episode_buffer()
    sim = lib._env.env.sim
    print(f"\n--- episode: task {args.task_id} init_state {init_id} | {task}\n"
          "waiting for the first motion key...", flush=True)
    return obs, sim.model._model, sim.data._data


init_id = args.init_state
obs, m, d = reset_episode(init_id)
viewer = mujoco.viewer.launch_passive(m, d, key_callback=on_key)
step, last_time, n_saved = 0, d.time, 0
dt = 1.0 / lib.control_freq  # real-time pacing: one env step = 0.05 s sim time
try:
    while viewer.is_running():
        if d.time < last_time - 1e-9:
            print("viewer reset detected (Backspace / Reset button) -> restarting properly", flush=True)
            state["cmd"] = "reset"
        cmd, state["cmd"] = state["cmd"], None
        if cmd:
            init_id = (init_id + (1 if cmd == "next" else 0)) % 50
            obs, m_new, d = reset_episode(init_id)
            if m_new is not m:  # hard reset rebuilt the MjModel -> reopen the viewer on it
                viewer.close()
                m = m_new
                viewer = mujoco.viewer.launch_passive(m, d, key_callback=on_key)
            step, last_time = 0, d.time

        if state["ended"] or state["paused"] or not state["started"]:
            viewer.sync()
            time.sleep(0.02)
            continue

        t0 = time.time()
        action = np.concatenate([state["vel"], [state["grip"]]])
        if ds is not None:
            ds.add_frame(to_frame(obs, action))  # observation paired with the action taken from it
        obs, _, terminated, _, info = lib.step(action)
        viewer.sync()
        last_time = d.time
        step += 1
        if info["is_success"] or terminated or step >= max_steps:
            state["ended"] = True
            if info["is_success"]:
                msg = f"SUCCESS in {step} steps"
                if ds is not None:
                    ds.save_episode()
                    n_saved += 1
                    msg += f" -> saved as episode {ds.meta.total_episodes - 1}"
            else:
                msg = f"FAIL ({'timeout' if step >= max_steps else 'done'}) after {step} steps -> discarded"
            print(msg + "\nF11 = same init state again, F12 = next init state, close window = quit", flush=True)
        time.sleep(max(0.0, dt - (time.time() - t0)))
finally:
    viewer.close()
    if ds is not None:
        if ds.has_pending_frames():
            ds.clear_episode_buffer()  # unfinished episode is never saved
        ds.finalize()
        print(f"saved {n_saved} new episodes this session, {ds.meta.total_episodes} total in {args.root}")
    vec.close()
