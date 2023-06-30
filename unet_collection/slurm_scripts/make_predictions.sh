#!/bin/bash

#SBATCH --partition=k40
#SBATCH --nodelist=wn122
#SBATCH -N 1
#SBATCH --output=/home/sciemala/thyroid_analysis/unet_collection/%j.out

# source ../.env/bin/activate

source ../.env/bin/activate
# python3 generate_predictions.py --model_path output/u2net_2d/samsung/test/model/model --dataset_type samsung --output_path prediction_outputs/u2net_2d_samsung_test.pickle

python3 generate_predictions.py --model_path output/u2net_2d/samsung/test/model/model --dataset_type ge --output_path prediction_outputs/u2net_2d_samsung_on_ge_test.pickle

# python3 generate_predictions.py --model_path output/u2net_2d/ge/test/model/model --dataset_type ge --output_path prediction_outputs/u2net_2d_ge_test.pickle

# python3 generate_predictions.py --model_path output/unet_3plus_2d/samsung/test/model/model --dataset_type samsung --output_path prediction_outputs/unet_3plus_2d_samsung_test.pickle

# python3 generate_predictions.py --model_path output/unet_3plus_2d/ge/test/model/model --dataset_type ge --output_path prediction_outputs/unet_3plus_2d_ge_test.pickle

# python3 generate_predictions.py --model_path output/transunet_2d/samsung/test/model/model --dataset_type samsung --output_path prediction_outputs/transunet_2d_samsung_test.pickle

# python3 generate_predictions.py --model_path output/transunet_2d/ge/test/model/model --dataset_type ge --output_path prediction_outputs/transunet_2d_ge_test.pickle