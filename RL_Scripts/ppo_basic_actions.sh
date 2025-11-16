#!/bin/bash
#SBATCH --job-name=ppo_craftax_actions        # Job name
#SBATCH --partition=bigbatch                  # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/rl_experiments/ppo_basic_actions_woodpick_all_seeds_2.out  # Combined output log

# Load your environment
source ~/.bashrc
conda activate SOTA

# Define your list of seeds
# seeds=(888 0 333 9 42)
# seeds=(2106 1 404 1408 506)
seeds=(888 0 333 9 42 2106 1 404 1408 506)

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo "Starting PPO training with seed: $seed"
    echo "=========================================="

    python ppo_basic_actions.py --ppo_seed "$seed"

    echo "Finished run for seed: $seed"
    echo
done

echo "All runs completed!"
