#!/bin/bash
#SBATCH --job-name=ppo_hierarchy_craftax        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/ppo_hierarchy_woodpick.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python ppo_hierarchy.py
echo "Done"

