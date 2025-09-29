#!/bin/bash
#SBATCH --job-name=gen_craftax          # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/start_pu_models_tuned.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA

python Skill_Learning/train_start_model_pulearning.py 