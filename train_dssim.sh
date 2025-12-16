#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:20:00

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

#SBATCH --exclude=gpu[002,012]

# run the application

# TRAIN CMD

#WANDB_DISABLE_SERVICE=True python train_dssim.py -s "/mnt/scratch/scasag/tmay/" --configs "/users/scasag/E-D3DGS2_audionet/arguments/may/default.py" --model_path "/mnt/scratch/scasag/audionet_dssim/" --expname "audionet_dssim" --images "gt_imgs" #-r 2

# RENDER CMD
####  SBATCH --exclude=gpu[003,006,026,027]

python render.py --model_path "/mnt/scratch/scasag/audionet_dssim/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS2_audionet/arguments/may/default.py" --images "gt_imgs" #--iteration 30000