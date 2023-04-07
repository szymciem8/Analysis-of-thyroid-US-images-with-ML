#!/bin/bash

#SBATCH --partition=k40
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/transunet_2d/%j.out
#SBATCH --nodelist=wn120

source ../.env/bin/activate
python train.py -m transunet_2d -b 8 -e 150 -p 30
