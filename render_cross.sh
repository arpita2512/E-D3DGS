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


#python render_cd.py --model_path "/mnt/scratch/scasag/obama1_L0/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/obama1.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/obama1/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/WRA1000_hu.npy"

#cd /mnt/scratch/scasag/obama1_L0/train/custom_WRA1000_hu/renders/

#ffmpeg -framerate 25 -i  %04d.jpg -i "/mnt/scratch/scasag/cross_driven/WRA1000.wav" obm_wra1000.mkv

#cd $HOME/E-D3DGS_bg


python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/ava1_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/adaline1_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/adaline2_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/autumn1_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/autumn2_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/hale1_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/jamie1_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/jamie2_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/ironrose1_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/ironrose2_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/michael1_hu.npy"

python render_cd.py --model_path "/mnt/scratch/scasag/macron_latest/" --configs "/users/scasag/E-D3DGS_bg/arguments/talkinghead/macron.py" --images "gt_imgs" --bg_path "/mnt/scratch/scasag/macron/bc.jpg" --aud_path "/mnt/scratch/scasag/cross_driven/michael2_hu.npy"
