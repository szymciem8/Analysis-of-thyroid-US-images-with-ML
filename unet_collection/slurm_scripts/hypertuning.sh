#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn124
#SBATCH -N 1

source ../.env/bin/activate
# jupyter nbconvert --execute --to notebook --inplace tuning/unet.ipynb
jupyter nbconvert --execute --to notebook --inplace tuning/u2net.ipynb
# jupyter nbconvert --execute --to notebook --inplace tuning/unet3plus.ipynb