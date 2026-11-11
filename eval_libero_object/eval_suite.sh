#!/usr/bin/env bash
# Step B: smolvla_libero on all 10 libero_object tasks x 10 episodes.
# One lerobot-eval process per task: libero sync envs are built eagerly for every task,
# so a single full-suite process would hold 30 MuJoCo envs in RAM. Per-task processes free RAM on exit.
# Resumable: tasks whose eval_info.json already exists are skipped.
set -o pipefail
export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
cd ~/lerobot
source .venv/bin/activate
OUT=outputs/eval/2026-10-12_smolvla_libero_object_suite_10x10
EXP=experiments/2026-10-12_smolvla_libero_object_suite_10x10
mkdir -p $EXP $OUT
pip freeze > $EXP/pip_freeze.txt
git rev-parse HEAD > $EXP/lerobot_commit.txt
( while true; do echo "$(date +%T) ram_used_mb=$(free -m | awk '/Mem/{print $3}') vram_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 30; done ) >> $EXP/resources.log &
LOGGER=$!
trap "kill $LOGGER" EXIT
echo "lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --policy.device=cuda --env.type=libero --env.task=libero_object --env.task_ids=[\$i] --env.init_states=true --eval.batch_size=3 --eval.use_async_envs=false --eval.n_episodes=10 --seed=1000 --output_dir=$OUT/task\$i   (i=0..9)" > $EXP/command.txt
for i in 0 1 2 3 4 5 6 7 8 9; do
  if [ -f $OUT/task$i/eval_info.json ]; then echo "$(date +%T) task$i already done, skip" >> $EXP/progress.log; continue; fi
  rm -rf $OUT/task$i
  echo "$(date +%T) task$i start" >> $EXP/progress.log
  START=$(date +%s)
  lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --policy.device=cuda --env.type=libero --env.task=libero_object \
    --env.task_ids=[$i] --env.init_states=true --eval.batch_size=3 --eval.use_async_envs=false --eval.n_episodes=10 \
    --seed=1000 --output_dir=$OUT/task$i > $EXP/run_task$i.log 2>&1
  RC=$?
  echo "$(date +%T) task$i exit=$RC wall_s=$(( $(date +%s) - START ))" >> $EXP/progress.log
  cp $OUT/task$i/eval_info.json $EXP/eval_info_task$i.json 2>/dev/null
done
echo "$(date +%T) ALL DONE" >> $EXP/progress.log
