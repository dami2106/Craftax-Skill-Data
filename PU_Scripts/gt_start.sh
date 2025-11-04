#!/bin/bash
#SBATCH --job-name=pu_start_craftax_gt        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/experiments_diss/start_pu_craftax_resnet_gt.out # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python Skill_Learning/train_start_model_pulearning.py --dir 'Traces/stone_pick_static' --skills_dirname 'groundTruth' \
--features_name 'resnet_features' --old_data_mode --save_dir 'pu_start_resnet_gt'
