#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:30:00

# set name of job
#SBATCH --job-name=macron

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --mem=100G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scasag@leeds.ac.uk

# run the application

# TRAIN CMD

#WANDB_DISABLE_SERVICE=True numactl --interleave=1-3 python train.py -s "/mnt/scratch/scasag/macron/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --model_path "/mnt/scratch/scasag/macron_latest/" --expname "macron_latest" --images "gt_imgs" --split_idx 7938 #-r 2

# RENDER CMD

numactl --interleave=1-3 python render.py --model_path "/mnt/scratch/scasag/macron_latest/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --split_idx 7938 --bg_path "/mnt/scratch/scasag/macron/bc.jpg" #--iteration 14000

# metrics

python metrics.py --model_path "/mnt/scratch/scasag/macron_latest/"