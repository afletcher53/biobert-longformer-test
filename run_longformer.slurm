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

# Print current working directory
echo "Current working directory: $(pwd)"

# List contents of current directory
echo "Directory contents:"
ls -l

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify activation
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Add local bin to PATH if necessary
if [ -d "$(pwd)/bin" ]; then
    echo "Adding local bin to PATH..."
    PATH=$(pwd)/bin:$PATH
fi

# Print updated PATH
echo "Updated PATH: $PATH"

# Output installed pip packages
echo "Installed pip packages:"
pip list

# Begin process
echo "Starting main process..."
python train-roberta-longformer.py