#!/bin/bash
#SBATCH --comment=af-biobert-longformer
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahafletcher1@sheffield.ac.uk
#SBATCH --output=log/%j.%x.out
#SBATCH --error=log/%j.%x.err

# Load necessary modules
module load Anaconda3/2019.07
module load cuDNN/8.0.4.30-CUDA-11.1.1    

# Initialize conda
eval "$(/usr/local/packages/live/eb/Anaconda3/2019.07/bin/conda shell.bash hook)"

# Activate your environment
conda activate longformer

# Print Python path and version for debugging
which python
python --version

# Print conda environment info
conda info

# Print current working directory
pwd

# List contents of current directory
ls -l

# Check if PyTorch is installed
conda list | grep torch

# print all the conda installs
conda list

# Begin process
python train-roberta-longformer.py