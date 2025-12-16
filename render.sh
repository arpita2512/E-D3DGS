#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:10:00

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

#SBATCH --exclude=gpu[002,007,019,026,028]

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=24

# run the application

# TRAIN CMD

python test_wandb.py -s "/mnt/scratch/scasag/tmay/" --configs "/users/scasag/E-D3DGS/arguments/may/default.py" --model_path "/mnt/scratch/scasag/test/" --expname "test" --images "gt_imgs" #-r 2

# RENDER CMD

#python render.py --model_path "/mnt/scratch/scasag/exp_lip_loss2/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS/arguments/may/default.py" --images "gt_imgs" --iteration 30000