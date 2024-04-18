#!/bin/bash
#
#SBATCH --job-name=train_Burgers
#SBATCH --output=/scratch/jcolen/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

echo "Starting"

cd /home/jcolen/microfluidics_neural_operators
env="default-pytorch-gpu-2.1.0"

betas=("1e-07" "1e-08")
beta=${betas[$SLURM_ARRAY_TASK_ID]}
echo "Running with beta ${beta}"
#python train_Burgers.py --beta $beta --epochs 500
#python continue_training.py --beta $beta --epochs 500

module load container_env pytorch-gpu/2.1.0

crun -p /home/jcolen/envs/${env} python train_Burgers.py \
    --beta ${beta} \
    --epochs 1000

#crun -p /home/jcolen/envs/${env} python train_dedalus.py \
#    --beta ${beta} \
#    --train_data data/burgers_dedalus.npy \
#    --epochs 1000
