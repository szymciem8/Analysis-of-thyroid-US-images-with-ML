#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn118
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/unet_3plus_2d/%j.out

source ../.env/bin/activate
python train.py -m unet_3plus_2d -b 16 -e 350 -p 50
