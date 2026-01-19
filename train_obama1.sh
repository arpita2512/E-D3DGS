#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=02:20:00

# set name of job
#SBATCH --job-name=obama1

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

WANDB_DISABLE_SERVICE=True numactl --interleave=1-3 python train.py -s "/mnt/scratch/scasag/obama1/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/obama1.py" --model_path "/mnt/scratch/scasag/obama1_final/" --expname "obama1_final" --images "gt_imgs" #-r 2

# RENDER CMD

numactl --interleave=1-3 python render.py --model_path "/mnt/scratch/scasag/obama1_test/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/obama1.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/obama1/bc.jpg" #--iteration 20000

# metrics

python metrics.py --model_path "/mnt/scratch/scasag/obama1_final/"
