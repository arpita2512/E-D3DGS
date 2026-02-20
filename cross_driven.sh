#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:15:00

# set name of job
#SBATCH --job-name=rndr_cd

#SBATCH --partition=gpu

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --mem=20G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=scasag@leeds.ac.uk

# run the application

# TRAIN CMD


for idn in adaline2 ironrose2 jamie1 shaheen
do
  python render_cd.py --model_path "/mnt/scratch/scasag/debbie_L0/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/debbie.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/debbie/bc.jpg" --aud_path /mnt/scratch/scasag/cross_driven/"$idn"_hu.npy
  
  cd /mnt/scratch/scasag/debbie_L0/train/custom_"$idn"_hu/renders/
  
  ffmpeg -framerate 25 -i  %04d.jpg -i /mnt/scratch/scasag/cross_driven/"$idn".wav deb_"$idn".mkv
  
  cd $HOME/E-D3DGS_bg
done
