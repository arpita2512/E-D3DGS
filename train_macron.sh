#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=02:30:00

# set name of job
#SBATCH --job-name=ed3dgs

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --mem=100G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scasag@leeds.ac.uk

#SBATCH --exclude=gpu[008,015,026]

# run the application

# TRAIN CMD

#WANDB_DISABLE_SERVICE=True python train.py -s "/mnt/scratch/scasag/macron/" --configs "/users/scasag/E-D3DGS2_audionet/arguments/may/default.py" --model_path "/mnt/scratch/scasag/macron_test/" --expname "macron_test" --images "gt_imgs" #-r 2

# RENDER CMD
####  SBATCH --exclude=gpu[003,006,026,027]

python render.py --model_path "/mnt/scratch/scasag/macron_test/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS2_audionet/arguments/may/default.py" --images "gt_imgs" #--iteration 14000