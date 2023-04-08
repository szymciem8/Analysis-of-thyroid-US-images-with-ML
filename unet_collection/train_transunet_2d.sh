#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn119
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/transunet_2d/%j.out

source ../.env/bin/activate
python train.py -m transunet_2d -b 2 -e 150 -p 30
