#!/bin/bash
#SBATCH --job-name=bc_cnn_craftax       # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/bc_cnn_craftax_all_2.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python Skill_Learning/behavioural_cloning_cnn.py --skill wood
echo "Done Wood"
python Skill_Learning/behavioural_cloning_cnn.py --skill wood_pickaxe
echo "Done Wood Pickaxe"
python Skill_Learning/behavioural_cloning_cnn.py --skill stone
echo "Done Stone"
python Skill_Learning/behavioural_cloning_cnn.py --skill stone_pickaxe
echo "Done Stone Pickaxe"
python Skill_Learning/behavioural_cloning_cnn.py --skill table
echo "Done Table"

