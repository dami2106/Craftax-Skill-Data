#!/bin/bash
#SBATCH --job-name=ppo_craftax_options_ompn      # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/rl_experiments/ppo_options_ompn_woodpick.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python ppo_skills.py --skill_list 0 1 2 3 4 --root Traces/stone_pick_static --bc_checkpoint_dir bc_checkpoints_ompn\
    --pca_model_path pca_models/pca_model_650.joblib --pu_start_models_dir pu_start_models_ompn --pu_end_models_dir pu_end_models_ompn --run_name ppo_options_ompn


echo "Done"
