#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn117
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/unet_2d/%j.out

source ../.env/bin/activate
python train.py -m unet_2d -b 12 -e 350 -p 40