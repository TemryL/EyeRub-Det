#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu
#SBATCH --mem 32G
#SBATCH --output=run.out

echo STARTING AT `date`

module load gcc cuda cudnn mvapich2
source ../venvs/hfi/bin/activate
srun python test.py
deactivate

echo FINISHED at `date`