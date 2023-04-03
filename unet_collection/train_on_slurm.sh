#!/bin/bash

#SBATCH --partition=k40

source ../.env/bin/activate
python train.py
