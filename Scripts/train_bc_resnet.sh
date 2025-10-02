#!/bin/bash
#SBATCH --job-name=bc_resnet_craftax       # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/bc_resnet_sp_static.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python Skill_Learning/bc_resnet.py --skill wood
echo "Done Wood"
python Skill_Learning/bc_resnet.py --skill wood_pickaxe
echo "Done Wood Pickaxe"
python Skill_Learning/bc_resnet.py --skill stone
echo "Done Stone"
python Skill_Learning/bc_resnet.py --skill stone_pickaxe
echo "Done Stone Pickaxe"
python Skill_Learning/bc_resnet.py --skill table
echo "Done Table"

