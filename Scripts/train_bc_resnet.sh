#!/bin/bash
#SBATCH --job-name=bc_resnet_craftax       # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/bc_resnet_sp_static_asot.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


    # parser = argparse.ArgumentParser(description="Train ResNet policy for a specific skill")
    # parser.add_argument("--skill", type=str, default="wood", help="Skill to train")
    # parser.add_argument("--dir", type=str, default="Traces/stone_pick_static", help="Dataset root")
    # parser.add_argument("--skills_name", type=str, default="groundTruth", help="Dataset root")

    # parser.add_argument("--image_dir_name", type=str, default="top_down_obs", help="Subdir with per-episode .npy images")
    # parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34"])
    # parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--epochs", type=int, default=150)
    # parser.add_argument("--lr", type=float, default=3e-4)
    # parser.add_argument("--weight_decay", type=float, default=3e-4)
    # parser.add_argument("--use_sampler", action="store_true", help="Use WeightedRandomSampler (else shuffle)")
    # parser.add_argument("--imagenet_norm", action="store_true", help="Use ImageNet mean/std (recommended for pretrained)")
    # args = parser.parse_args()

for skill in 0 1 2 3 4; do
    python Skill_Learning/bc_resnet.py --skill "$skill" --dir 'Traces/stone_pick_static' \
    --image_dir_name 'top_down_obs' --backbone 'resnet34' --skills_name 'asot_predicted' --save_dir 'bc_checkpoints_asot'
    echo "Done $skill"
done

