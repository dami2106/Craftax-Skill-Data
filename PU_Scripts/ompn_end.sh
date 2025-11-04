#!/bin/bash
#SBATCH --job-name=pu_end_craftax_ompn
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/dharvey/HiSD/experiments_diss/end_pu_craftax_resnet_ompn.out
# Load your environment

source ~/.bashrc
conda activate SOTA


python Skill_Learning/train_end_model_pulearning.py --dir 'Traces/stone_pick_static' --skills_dirname 'ompn_skills' \
--features_name 'resnet_features' --old_data_mode --save_dir 'pu_end_resnet_ompn'
