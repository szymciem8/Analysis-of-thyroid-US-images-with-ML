#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn121
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/swin_unet_2d/slurm_logs/%j.out

source ../.env/bin/activate

python train.py --model swin_unet_2d --dataset_type samsung --batch_size 8 --nfold 4 --fold_id 0 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320
python train.py --model swin_unet_2d --dataset_type samsung --batch_size 8 --nfold 4 --fold_id 1 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320
python train.py --model swin_unet_2d --dataset_type samsung --batch_size 8 --nfold 4 --fold_id 2 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320
python train.py --model swin_unet_2d --dataset_type samsung --batch_size 8 --nfold 4 --fold_id 3 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320


python train.py --model swin_unet_2d --dataset_type ge --batch_size 8 --nfold 4 --fold_id 0 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320
python train.py --model swin_unet_2d --dataset_type ge --batch_size 8 --nfold 4 --fold_id 1 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320
python train.py --model swin_unet_2d --dataset_type ge --batch_size 8 --nfold 4 --fold_id 2 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320
python train.py --model swin_unet_2d --dataset_type ge --batch_size 8 --nfold 4 --fold_id 3 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320


# Test runs
# python train.py --model swin_unet_2d --dataset_type samsung --batch_size 8 --nfold 4 --fold_id 0 --loss_type custom_focal_tversky --epochs 5000 --patiance 60 --input_size 320 --custom_directory test_arch