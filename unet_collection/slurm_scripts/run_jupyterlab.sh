#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn119
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/%j.out

ssh -N -f -R 8080:localhost:8080 fs0
source .env/bin/activate
jupyter-lab --no-browser --port=8080
