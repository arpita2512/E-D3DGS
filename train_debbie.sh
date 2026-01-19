#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=02:00:00

# set name of job
#SBATCH --job-name=debbie

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

WANDB_DISABLE_SERVICE=True numactl --interleave=1-3 python train.py -s "/mnt/scratch/scasag/debbie/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/debbie.py" --model_path "/mnt/scratch/scasag/debbie_finalL3/" --expname "debbie_finalL3" --images "gt_imgs" #-r 2

# RENDER CMD

numactl --interleave=1-3 python render.py --model_path "/mnt/scratch/scasag/debbie_finalL3/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/debbie.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/debbie/bc.jpg" #--iteration 20000

# metrics

python metrics.py --model_path "/mnt/scratch/scasag/debbie_finalL3/"