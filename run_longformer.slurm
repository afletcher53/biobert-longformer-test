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
source /usr/local/anaconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate longformer

# Print Python path and version for debugging
which python
python --version

# Print conda environment info
conda info

# Begin process
python train-bio-longformer.py