# Notes

Things I learned or ran into along the way, in the order they came up.

## Setup and first runs

- A short smoke test (task 0, 2 episodes) only showed that the pipeline works end to end. Two episodes say nothing about the success rate.
- With `n_action_steps=1` the policy predicts a 50-step action chunk but uses only its first step and replans every step. That is why evaluation runs at about 1 step per second.
- 5 or 10 episodes on one task do not narrow the confidence interval much. The meaningful number is the suite average over 100 episodes.
- torchcodec could not find FFmpeg at first, and LeRobot fell back to pyav. After installing FFmpeg, video decoding goes through torchcodec.

## Full suite evaluation

- In sync mode LIBERO builds the environments of every task at the start, so I ran one process per task. Each process frees its memory when it exits.
- The seed does not change the initial scenes. Episode i of a task always uses init state i, so another seed replays the same 10 scenes and only changes the policy sampling noise.
- Task 0 (3/10) and task 7 (5/10) are the weakest. I have not reviewed their videos yet.

## Live viewer

- The MuJoCo viewer already uses every letter key and most symbols as shortcuts, so the tools use only F9 to F12.
- The Reset button in the viewer calls `mj_resetData`. It puts every object back at its XML default pose, where the objects overlap and fly apart. The script detects the simulation time going backwards and does a proper LIBERO reset instead.
- One SmolVLA inference takes about 800 ms on the RTX 4060, which is slower than I expected. With `n_action_steps=10` the live view runs at about 7 steps per second. I want to time the parts of the model (VLM prefix, denoising steps, dtype) later.
- A viewer that crashes without `viewer.close()` leaves the process hanging, so the scripts close it in a `finally` block.

## Teleop

- The viewer only reports key presses, never releases. Motion is therefore sticky: each press adds a speed level until the opposite key or Num 5 stops it.
- `LeRobotDataset(repo_id, root=...)` quietly downloads missing files from the Hub. Asking for `episodes=[0]` once pulled a 37.5 MB video of a different task. Setting `HF_HUB_OFFLINE=1` and checking that files are local first avoids this.

## Open question: image colors

The frames I render look darker and more saturated than the dataset frames. Orientation and geometry match. The policy was trained on the dataset images, so this could be a distribution shift and part of the gap to the paper. I have only noted it so far.

## Dataset visualization

The rerun desktop viewer does not open on WSL2 (`Adapter does not support drawing to texture format R32Float`). Running `rerun --web-viewer <file>.rrd` opens it in the Windows browser instead.
