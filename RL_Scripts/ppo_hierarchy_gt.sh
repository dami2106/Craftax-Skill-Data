#!/bin/bash
#SBATCH --job-name=ppo_craftax_hierarchy_gt      # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/rl_experiments/ppo_hierarchy_groundtruth_woodpick_777.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python ppo_hierarchy.py --skill_list wooden_pickaxe stone_pickaxe wood stone table --root Traces/stone_pick_static --bc_checkpoint_dir bc_checkpoints_gt\
    --pca_model_path pca_models/pca_model_650.joblib --pu_start_models_dir pu_start_models_gt --pu_end_models_dir pu_end_models_gt --run_name ppo_hierarchy_groundtruth --hierarchy_dir Traces/stone_pick_static/hierarchy_data/ground_truth_hierarchy --ppo_seed 777


echo "Done"
