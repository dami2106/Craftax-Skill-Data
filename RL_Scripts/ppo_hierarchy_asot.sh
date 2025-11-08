#!/bin/bash
#SBATCH --job-name=ppo_craftax_hierarchy_asot      # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/experiments_diss/ppo_hierarchy_asot_woodpick_all_seeds.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


# Define your list of seeds
# seeds=(888 0 333 9 42)
# seeds=(2106 1 404 1408 506)

seeds=(888 0 333 9 42 2106 1 404 1408 506 777 123 1111 2024 999)



# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo "Starting PPO training with seed: $seed"
    echo "=========================================="

    # python ppo_hierarchy.py --skill_list 0 1 2 3 4 --root Traces/stone_pick_static --bc_checkpoint_dir bc_checkpoints_asot\
    # --pca_model_path pca_models/pca_model_650.joblib --pu_start_models_dir pu_start_models_asot --pu_end_models_dir pu_end_models_asot --run_name ppo_hierarchy_asot --hierarchy_dir Traces/stone_pick_static/hierarchy_data/asot_predicted_hierarchy --symbol_map "" --ppo_seed "$seed"

    python ppo_hierarchy.py --skill_list 0 1 2 3 4 --root Traces/stone_pick_static --bc_checkpoint_dir bc_checkpoints_asot\
    --pca_model_path pca_models/pca_model_650.joblib --pu_start_models_dir BLANK --pu_end_models_dir BLANK --run_name ppo_hierarchy_asot_${seed} --hierarchy_dir Traces/stone_pick_static/hierarchy_data/asot_predicted_hierarchy --symbol_map "" --ppo_seed "$seed"


    echo "Finished run for seed: $seed"
    echo
done

echo "All runs completed!"


