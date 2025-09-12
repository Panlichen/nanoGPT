#!/bin/bash
#SBATCH --gpus=1
module load miniforge3/24.11
source activate nanogpt310

python sample.py --out_dir=out-shakespeare-char