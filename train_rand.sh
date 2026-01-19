#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=01:50:00

# set name of job
#SBATCH --job-name=rand

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --mem=70G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scasag@leeds.ac.uk

# run the application

# TRAIN CMD

WANDB_DISABLE_SERVICE=True numactl --interleave=1-3 python train_rand.py -s "/mnt/scratch/scasag/rand/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/rand.py" --model_path "/mnt/scratch/scasag/rand_finalL2/" --expname "rand_finalL2" --images "gt_imgs" #-r 2

# RENDER CMD

numactl --interleave=1-3 python render.py --model_path "/mnt/scratch/scasag/rand_finalL2/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/rand.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/rand/bc.jpg" #--iteration 20000

# metrics

#python metrics.py --model_path "/mnt/scratch/scasag/rand_final/"
