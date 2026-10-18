#!/bin/bash

#SBATCH --job-name=crax_crpo_proprio_concat_push_L1
#SBATCH --output=push_L1_proprio_concat_%j.txt
#SBATCH --partition=gpu_a100 # Choose a partition that has GPU
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

uv sync --all-extras
# Execute the script or command
uv run python -m training.train_env --env_name safe_push_point --alg crpo --difficulty 1 --num_timesteps 300000000 --seeds 0 1 2 3 4 --vision --vision_obs_mode="pixels+state" --env_kwargs '{"include_goal_lidar": false, "include_hazard_lidar": false, "include_goal_comp": false, "include_hazard_comp": false, "include_block_obs": false}' --vision_layer_norm --vision_early_fusion concat --wandb_group safe_push_L1_proprio_concat --wandb_tags vision_proprio concat safe_push crpo level1
