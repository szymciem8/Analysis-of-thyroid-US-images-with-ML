#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn119
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/unet_2d/slurm_logs/%j.out

source ../.env/bin/activate
python train.py --model unet_2d --dataset_type samsung --batch_size 16 --loss_type custom_focal_tversky --epochs 350 --patiance 60 --test True