#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:5:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000
#SBATCH --output=run.out

echo STARTING AT `date`

module load gcc cuda cudnn mvapich2
source ../venvs/hfi/bin/activate
srun python train_supervised.py 'exp'
deactivate

echo FINISHED at `date`