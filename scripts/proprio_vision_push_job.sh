#!/bin/bash

#SBATCH --job-name=crax_crpo_pixel_state_safe_push
#SBATCH --output=safe_push_concat_proprio_vision_%j.txt
#SBATCH --partition=gpu_a100 # Choose a partition that has GPU
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

uv sync --all-extras
# Execute the script or command
uv run python -m training.train_env --env_name safe_push_point --alg crpo --difficulty 1 --vision --vision_obs_mode="pixels+state" --env_kwargs '{"include_goal_lidar": false, "include_hazard_lidar": false, "include_goal_comp": false, "include_hazard_comp": false}' --vision_layer_norm --vision_early_fusion concat --wandb_tags concat safe_push crpo vision_proprio
