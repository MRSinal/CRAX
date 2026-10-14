#!/bin/bash

#SBATCH --job-name=crax_crpo_state_safe_goal
#SBATCH --output=state_job_output_%j.txt
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
uv run python -m training.train_env --env_name safe_goal_point --alg crpo --difficulty 1 --env_kwargs '{"include_goal_lidar": true, "include_hazard_lidar": true, "include_goal_comp": true, "include_hazard_comp": true}'
