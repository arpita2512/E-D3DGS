#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=02:00:00

# set name of job
#SBATCH --job-name=ed3dgs

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --mem=80G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scasag@leeds.ac.uk

# run the application

# TRAIN CMD

python train2.py -s "/mnt/scratch/scasag/aligned_input/" --configs "/users/scasag/E-D3DGS/arguments/mead/default.py" --model_path "/mnt/scratch/scasag/align_op_fifteen/" --expname "mead" #-r 2

# RENDER CMD

#python render.py --model_path "/mnt/scratch/scasag/align_op/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS/arguments/mead/default.py" #--iteration 60000
