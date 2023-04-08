#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn116
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/output/u2net_2d/%j.out

source ../.env/bin/activate
python train.py -m u2net_2d -b 8 -e 150 -p 30
