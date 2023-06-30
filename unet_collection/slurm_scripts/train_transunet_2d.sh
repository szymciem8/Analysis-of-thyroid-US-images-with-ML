#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn122
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/transunet_2d/slurm_logs/%j.out

# source ../.env/bin/activate

source ../.env2/bin/activate
python train.py --model transunet_2d --dataset_type samsung --batch_size 8 --loss_type custom_focal_tversky --epochs 5000 --patiance 100 --input_size 320 --custom_directory size_320_model_1