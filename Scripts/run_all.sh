#!/bin/bash
#SBATCH --job-name=gen_craftax          # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/generate_data_sp_static.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA

echo "Starting data generation..."

# python get_truth_stats.py
# echo "Truth stats generated."



python train_pca_model.py --components 650
# python train_pca_model.py --components 750
# python train_pca_model.py --components 1000
# python train_pca_model.py --components 2000
# echo "PCA model trained."


python get_pca_features.py --components 650
# python get_pca_features.py --components 750
# python get_pca_features.py --components 1000
# python get_pca_features.py --components 2000
# echo "PCA features extracted."


# python get_clip_features.py
# echo "CLIP features extracted."


# python get_resnet_features.py
# echo "ResNet features extracted."