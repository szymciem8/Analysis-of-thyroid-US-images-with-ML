#!/bin/bash

while getopts ":v:m:" opt; do
  case $opt in
    v) version="$OPTARG";;
    m) model="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2;;
  esac
done


sbatch train_model.sh -m model -logpah