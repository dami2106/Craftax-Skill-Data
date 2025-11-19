#!/bin/bash
#SBATCH --job-name=bc_recurent_craftax_compile      # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/bc_recurent_craftax_compile.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA

for skill in 0 1 2 3 4; do
    python Skill_Learning/bc_pca_recurrent.py --skill "$skill" --dir 'Traces/stone_pick_static' \
    --skills_name 'compile_skills' --save_dir 'bc_checkpoints_pca_gru_compile' --grid
    echo "Done $skill"
done

