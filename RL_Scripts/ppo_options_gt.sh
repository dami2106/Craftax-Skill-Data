#!/bin/bash
#SBATCH --job-name=ppo_craftax_options_gt      # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/experiments_diss/ppo_options_groundtruth_all_seeds.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


# Define your list of seeds
# seeds=(888 0 333 9 42)
# seeds=(2106 1 404 1408 506)
# seeds=(888)
seeds=(888 0 333 9 42 2106 1 404 1408 506 777 123 1111 2024 999)


# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo "Starting PPO training with seed: $seed"
    echo "=========================================="

#     python ppo_skills.py --skill_list wooden_pickaxe stone_pickaxe wood stone table --root Traces/stone_pick_static --bc_checkpoint_dir bc_checkpoints_gt\
#    --pu_start_models_dir pu_start_resnet_gt --pu_end_models_dir pu_end_resnet_gt --run_name ppo_options_groundtruth_$seed --ppo_seed "$seed"
    python ppo_skills.py --skill_list wooden_pickaxe stone_pickaxe wood stone table --root Traces/stone_pick_static --bc_checkpoint_dir bc_checkpoints_gt\
   --pu_start_models_dir BLANK --pu_end_models_dir BLANK --run_name ppo_options_groundtruth_no_pu_$seed --ppo_seed "$seed"

    echo "Finished run for seed: $seed"
    echo
done

echo "All runs completed!"

