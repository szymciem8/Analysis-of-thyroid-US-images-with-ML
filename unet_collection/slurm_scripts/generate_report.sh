#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn120
#SBATCH -N 1

source ../.env/bin/activate
jupyter nbconvert --execute --to notebook --inplace model_tester.ipynb