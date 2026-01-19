#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=02:00:00

# set name of job
#SBATCH --job-name=may

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --mem=75G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scasag@leeds.ac.uk

# run the application

# TRAIN CMD

WANDB_DISABLE_SERVICE=True numactl --interleave=1-3 python train.py -s "/mnt/scratch/scasag/tmay/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/may.py" --model_path "/mnt/scratch/scasag/may_dsdrdodc/" --expname "may_dsdrdodc" --images "gt_imgs" --split_idx 5520 #-r 2

# RENDER CMD

numactl --interleave=1-3 python render.py --model_path "/mnt/scratch/scasag/may_dsdrdodc/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/may.py" --images "gt_imgs"  --split_idx 5520 #--iteration 20000

# metrics

python metrics.py --model_path "/mnt/scratch/scasag/may_dsdrdodc/"
