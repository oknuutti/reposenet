#!/bin/bash
#SBATCH --time=0-00:30:00		# estimated excution time (~80s/epoch)
#SBATCH --mem=12G     			# memory needed (~2Gb/core)
#SBATCH --gres=gpu:1			# GPUs needed
#SBATCH -c 4				# CPUs needed (~9 per one gpu)
#SBATCH --constraint='pascal|volta'	# exclude the slowest GPUs

## copy image data to local drive
mkdir /tmp/$SLURM_JOB_ID
trap "rm -r /tmp/$SLURM_JOB_ID; exit" TERM EXIT
tar -xf $WRKDIR/data/stmaryschurch.tar -C /tmp/$SLURM_JOB_ID

## start to process
cd $WRKDIR/densepose/src
module load anaconda3
source activate $WRKDIR/conda/envs/densepose
srun python3 main.py -d /tmp/$SLURM_JOB_ID/cambridge/StMarysChurch --cache $WRKDIR/data/models \
                     -a googlenet -o adam --lr 1e-2 --wd 0.1 --do 0.5 -b 64 --epochs 50 -j 4 --tf 3
