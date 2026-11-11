#!/usr/bin/env bash
# Launcher for live_policy.py: sets the WSLg/GPU render env (non-interactive shells skip .bashrc).
# Usage: bash interactive/run_live.sh --task-id 0 --init-state 0   (all args go to live_policy.py)
export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
cd "${LEROBOT_DIR:-$HOME/lerobot}" && source .venv/bin/activate
python interactive/live_policy.py "$@" 2>&1 | grep --line-buffered -v "robosuite\|Downloading\|^$"
