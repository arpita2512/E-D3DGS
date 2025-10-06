#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=01:30:00

# set name of job
#SBATCH --job-name=ed3dgs

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --mem=50G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scasag@leeds.ac.uk

# run the application

# TRAIN CMD

python train.py -s "/mnt/scratch/scasag/vrig-chicken/" --configs "/users/scasag/E-D3DGS/arguments/hypernerf/vrig-chicken.py" --model_path "/mnt/scratch/scasag/chick_op/" --expname "chick" -r 2

# RENDER CMD

#python render.py --model_path "/mnt/scratch/scasag/chicken_op/"  --skip_train --configs "/users/scasag/E-D3DGS/arguments/hypernerf/vrig-chicken.py"
