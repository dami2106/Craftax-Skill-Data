#!/bin/bash
#SBATCH --job-name=bc_recurent_craftax_gt      # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/bc_recurent_craftax_gt.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA

# wood
# wood_pickaxe
# stone
# stone_pickaxe
# table

for skill in wood wood_pickaxe stone stone_pickaxe table; do
    python Skill_Learning/bc_pca_recurrent.py --skill "$skill" --dir 'Traces/stone_pick_static' \
    --skills_name 'groundTruth' --save_dir 'bc_checkpoints_pca_gru_gt' --grid
    echo "Done $skill"
done

