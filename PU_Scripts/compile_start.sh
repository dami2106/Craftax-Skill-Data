#!/bin/bash
#SBATCH --job-name=pu_start_craftax_compile
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/dharvey/HiSD/experiments_diss/start_pu_craftax_resnet_compile.out
# Load your environment

source ~/.bashrc
conda activate SOTA


python Skill_Learning/train_start_model_pulearning.py --dir 'Traces/stone_pick_static' --skills_dirname 'compile_skills' \
--features_name 'resnet_features' --old_data_mode --save_dir 'pu_start_resnet_compile'
