#!/bin/bash

#SBATCH --job-name=crax_crpo_pixel_push_L2
#SBATCH --output=push_L2_pixel_%j.txt
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
uv run python -m training.train_env --env_name safe_push_point --alg crpo --difficulty 2 --num_timesteps 300000000 --seeds 0 1 2 3 4 --vision --vision_obs_mode="pixels" --env_kwargs '{"include_goal_lidar": false, "include_hazard_lidar": false, "include_goal_comp": false, "include_hazard_comp": false}' --vision_layer_norm --wandb_group safe_push_L2_pixel --wandb_tags vision_pixel pixel safe_push crpo level2
