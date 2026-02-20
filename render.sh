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

#SBATCH --mem=30G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scasag@leeds.ac.uk

# run the application

# TRAIN CMD

python render.py --model_path "/mnt/scratch/scasag/debbie_L0/"  --skip_video --skip_train --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/debbie.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/debbie/bc.jpg"