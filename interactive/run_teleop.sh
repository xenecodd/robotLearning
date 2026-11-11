#!/usr/bin/env bash
# Keyboard teleop + demo recording in the MuJoCo viewer. Args go to teleop_record.py (see its docstring).
export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
cd "${LEROBOT_DIR:-$HOME/lerobot}" && source .venv/bin/activate
python interactive/teleop_record.py "$@" 2>&1 | grep --line-buffered -v "robosuite\|Downloading\|^$"
