# Learning robot learning with LeRobot and LIBERO

This is a learning project. I use it to learn how robot learning policies are evaluated and trained, with [LeRobot](https://github.com/huggingface/lerobot) and the [LIBERO](https://libero-project.github.io/) benchmark in MuJoCo. It is early work. The list at the bottom says what is done and what is not done yet.

Hardware: RTX 4060 8 GB on WSL2 (Ubuntu 24.04), rendering through EGL and Mesa d3d12.

## 1. Evaluating a pretrained policy

I ran the pretrained SmolVLA checkpoint `HuggingFaceVLA/smolvla_libero` on all 10 tasks of the `libero_object` suite, 10 episodes per task, 100 episodes in total.

| task | success | % | Wilson 95% CI |
|---|---|---|---|
| 0 | 3/10 | 30 | 11–60 |
| 1 | 8/10 | 80 | 49–94 |
| 2 | 9/10 | 90 | 60–98 |
| 3 | 9/10 | 90 | 60–98 |
| 4 | 9/10 | 90 | 60–98 |
| 5 | 10/10 | 100 | 72–100 |
| 6 | 9/10 | 90 | 60–98 |
| 7 | 5/10 | 50 | 24–76 |
| 8 | 5/10 | 50 | 24–76 |
| 9 | 6/10 | 60 | 31–83 |
| **total** | **73/100** | **73.0** | **63.6–80.7** |

The SmolVLA paper reports 93 to 96% on this suite. My 73% is clearly below that, and I have not found the cause yet. Two things I checked or found:

- The checkpoint is not identical to any model size in the paper. I counted its weights: 0.605B parameters with all 32 VLM layers. All paper sizes score 93 to 96% on this suite, so the comparison holds either way.
- Other users also report scores below the paper ([lerobot#2354](https://github.com/huggingface/lerobot/issues/2354)). A MuJoCo collision fix from version 3.4 changes how some initial scenes settle ([lerobot#4390](https://github.com/huggingface/lerobot/issues/4390)). I used MuJoCo 3.8.1, so this is my strongest open guess. I have not tested it.

The control mode was `relative` (delta end effector actions), which matches how the LIBERO demos were recorded.

Files: [`eval_libero_object/`](eval_libero_object). `eval_suite.sh` runs one `lerobot-eval` process per task, because a single process kept the environments of all tasks in memory. `aggregate.py` builds the table above from the per-task `eval_info` files.

## 2. Tools for watching and driving the robot

[`interactive/live_policy.py`](interactive/live_policy.py) runs the policy live in the MuJoCo viewer. You can pause it, step it, restart a scene and push objects with the mouse while the policy acts. It uses the same observation to action path as `lerobot-eval`.

[`interactive/teleop_record.py`](interactive/teleop_record.py) drives the arm with the numpad and saves successful episodes in the same format as the LIBERO dataset. I checked that its robot state matches a dataset episode to the millimetre. I have not yet recorded a real demo set with it.

## 3. Preparing data for a behavior cloning policy

Next I want to train a small policy of my own on one task, LIBERO object task 0 ("pick up the alphabet soup and place it in the basket"). [`bc_data_prep/explore_dataset.ipynb`](bc_data_prep/explore_dataset.ipynb) prepares the data:

- loads the 44 demos of that task (6,867 frames) from `lerobot/libero`
- splits them into train and validation sets by episode, so frames of one demo never end up in both
- computes action mean and std on the train set only ([`action_stats.json`](bc_data_prep/action_stats.json))
- resizes images to 128x128, normalizes the actions and caches both to disk as tensors
- wraps the cache in a PyTorch `Dataset`

The split is in [`split.json`](bc_data_prep/split.json). [`viz_episode.py`](bc_data_prep/viz_episode.py) writes one episode to a [rerun](https://rerun.io/) file, to look at the camera, state and actions on one timeline.

## Status

Done:
- SmolVLA evaluation on `libero_object`, 100 episodes
- live policy viewer and numpad teleop tool
- data preparation for behavior cloning on task 0

Not done yet:
- training my own policy (the model is the next step)
- recording my own demonstrations with the teleop tool
- finding the cause of the gap to the paper
- anything on a real robot

## Setup notes

- LeRobot at commit `b9cb121c` (see [`env/`](env) for the full package list).
- On WSL2, set `MUJOCO_GL=egl GALLIUM_DRIVER=d3d12` or MuJoCo renders on the CPU at about half the speed.
- `lerobot/smolvla_base` does not fit LIBERO: its camera names and state and action sizes differ. Use a checkpoint fine-tuned on LIBERO.
- With async environments each env starts a process of about 1 GB, which ran out of memory on WSL at batch size 5. Synchronous envs with batch size 3 worked.
- The run scripts expect the LeRobot checkout in `~/lerobot`, or in `$LEROBOT_DIR` if set.

More notes from the work are in [NOTES.md](NOTES.md).
