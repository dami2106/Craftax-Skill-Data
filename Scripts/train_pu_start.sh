#!/bin/bash
#SBATCH --job-name=pu_start_compile        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/rl_experiments/start_pu_models_compile_skills.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python Skill_Learning/train_start_model_pulearning.py --dir 'Traces/stone_pick_static' --skills_dirname 'compile_skills' \
--features_name 'pca_features_650' --old_data_mode --save_dir 'pu_start_models_compile'
