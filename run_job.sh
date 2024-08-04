#!/bin/bash
SBATCH --comment=af-biobert-longformer
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04-00:00:00
#SBATCH --mail-type=ALL
SBATCH --mail-user=ahafletcher1@sheffield.ac.uk
#SBATCH --output=log/%j.%x.out
#SBATCH --error=log/%j.%x.err

module load Anaconda3/2022.10
module load cuDNN/8.7.0.84-CUDA-11.8.0

# init env
source .venv/bin/activate
PATH=$(pwd)/bin:$PATH
# begin process
python train-bio-longformer.py