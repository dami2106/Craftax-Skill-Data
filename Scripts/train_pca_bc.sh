#!/bin/bash

for item in \
0 \
1 \
2 \
3 \
4
do
    echo "$item"
    python Skill_Learning/bc_logistc.py --skill "$item" --dir 'Traces/stone_pick_static' \
    --skills_name 'ompn_skills' --feature_dir_name 'pca_features_650' --save_dir 'bc_checkpoints_pca_ompn' --grid
    echo "Done $item"
done
