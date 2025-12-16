#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:30:00

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

#SBATCH --exclude=gpu[007,014]

# run the application

# TRAIN CMD

#WANDB_DISABLE_SERVICE=True python train_mouth.py -s "/mnt/scratch/scasag/tmay/" --configs "/users/scasag/E-D3DGS2_audionet/arguments/may/mouth.py" --model_path "/mnt/scratch/scasag/may_mouth_crop/" --expname "may_mouth_crop" --images "gt_imgs" #-r 2

# RENDER CMD
####  SBATCH --exclude=gpu[003,006,026,027]

python render_mouth.py --model_path "/mnt/scratch/scasag/may_mouth_crop/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS2_audionet/arguments/may/mouth.py" --images "gt_imgs" #--iteration 14000