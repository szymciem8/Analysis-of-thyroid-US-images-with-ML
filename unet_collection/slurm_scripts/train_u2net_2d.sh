#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn121
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/u2net_2d/slurm_logs/%j.out

source ../.env/bin/activate
python train.py --model u2net_2d --dataset_type ge --batch_size 8 --loss_type custom_focal_tversky --epochs 5000 --patiance 100 --input_size 320 --custom_directory size_320_model_1
