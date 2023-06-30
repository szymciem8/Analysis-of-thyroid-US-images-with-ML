#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn118
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/unet_2d/slurm_logs/%j.out

source ../.env/bin/activate
# python train.py --model unet_2d --input_size 256 --dataset_type samsung --batch_size 16 --loss_type focal_tversky_loss --epochs 1000 --patiance 60 --test True
# python train.py --model unet_2d --input_size 512 --dataset_type samsung --batch_size 16 --loss_type focal_tversky_loss --epochs 1000 --patiance 60 --custom_directory 512_1
# python train.py --model unet_2d --input_size 512 --dataset_type samsung --batch_size 16 --loss_type focal_tversky_loss --epochs 1000 --patiance 60 --custom_directory 512_2
# python train.py --model unet_2d --input_size 512 --dataset_type samsung --batch_size 16 --loss_type focal_tversky_loss --epochs 1000 --patiance 60 --custom_directory 512_3

# Additional tests
# python train.py --model unet_2d --input_size 256 --dataset_type samsung --batch_size 16 --loss_type dice --epochs 1000 --patiance 60 --custom_directory dice_test

python train.py --model unet_2d --dataset_type ge --batch_size 8 --loss_type custom_focal_tversky --epochs 5000 --patiance 100 --input_size 320 --custom_directory size_320_model_1