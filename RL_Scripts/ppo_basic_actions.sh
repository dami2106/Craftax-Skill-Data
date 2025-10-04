#!/bin/bash
#SBATCH --job-name=ppo_craftax_actions        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/rl_experiments/ppo_basic_actions_woodpick_777.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python ppo_basic_actions.py --ppo_seed 777
echo "Done"

