"""Visualize one LIBERO episode with rerun.

Usage (inside the lerobot checkout, with .venv active):
    python viz_episode.py --episode 908            # writes viz/ep908.rrd
    python viz_episode.py --episode 908 --spawn    # also opens the viewer (needs WSLg)

To open the .rrd file: `rerun viz/ep908.rrd`, or `rerun --web-viewer viz/ep908.rrd` on WSL2.
Timeline "frame" (0.1 s = 1 frame): scrubbing it moves the cameras, action and state together.
"""

import argparse
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # set before importing lerobot, blocks silent downloads

import numpy as np
import rerun as rr

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.expanduser("~/lerobot/data/libero_task0")
OUT_DIR = os.path.expanduser("~/lerobot/viz")


def to_uint8_hwc(img):
    """CHW float32 [0,1] torch tensor -> HWC uint8 numpy."""
    return (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, required=True, help="episode_index etiketi (e.g. 908)")
    p.add_argument("--root", default=ROOT)
    p.add_argument("--spawn", action="store_true", help="open the rerun viewer")
    args = p.parse_args()

    ds = LeRobotDataset("lerobot/libero", root=args.root, episodes=[args.episode])
    cam_keys = [k for k in ds.meta.features if k.startswith("observation.images")]
    print(f"episode {args.episode}: {len(ds)} frame, kameralar: {cam_keys}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"ep{args.episode}.rrd")
    rr.init(f"libero_ep{args.episode}", spawn=args.spawn)
    rr.save(out)

    for i in range(len(ds)):
        item = ds[i]
        rr.set_time("frame", sequence=i)
        for k in cam_keys:
            rr.log(k.replace(".", "/"), rr.Image(to_uint8_hwc(item[k])))
        for j, v in enumerate(item["action"].tolist()):
            rr.log(f"action/{j}", rr.Scalars(v))
        if "observation.state" in item:
            for j, v in enumerate(item["observation.state"].tolist()):
                rr.log(f"state/{j}", rr.Scalars(v))

    rr.log("task", rr.TextLog(item["task"]), static=True)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
