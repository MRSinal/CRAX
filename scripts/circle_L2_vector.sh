#!/bin/bash

#SBATCH --job-name=crax_crpo_vector_circle_L2
#SBATCH --output=circle_L2_vector_%j.txt
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
uv run python -m training.train_env --env_name safe_circle_point --alg crpo --difficulty 2 --num_timesteps 300000000 --seeds 0 1 2 3 4 --env_kwargs '{"include_hazard_lidar": true, "hazard_compass_k": 8, "circle_lidar": true}' --wandb_group safe_circle_L2_vector --wandb_tags vector safe_circle crpo level2
