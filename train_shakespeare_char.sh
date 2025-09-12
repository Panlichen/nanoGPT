#!/bin/bash
#SBATCH --gpus=1
module load miniforge3/24.11
source activate nanogpt310

python train.py config/train_shakespeare_char.py