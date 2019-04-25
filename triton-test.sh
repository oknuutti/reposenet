#!/bin/bash
#SBATCH --time=0-00:15:00		# 15 mins
#SBATCH --mem=10G     			# 10GB of memory
#SBATCH --gres=gpu:1			# one GPU
#SBATCH -c 2					# two CPUs
#SBATCH --constraint='pascal|volta'		# exclude the slowest GPUs

cd $WRKDIR/densepose/src
module load anaconda3
source activate $WRKDIR/conda/densepose
srun python3 main.py -d $WRKDIR/data/cambridge -a densenet121 -o adam --lr 1e-4 --wd 0.5 -b 64 --epochs 1
