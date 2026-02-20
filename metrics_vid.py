#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

import cv2

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    for fname in sorted(os.listdir(gt_dir)):
        gt = Image.open(f"{gt_dir}/{fname}")
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
    
    
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(f"{renders_dir}/{fname}")        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())

    return renders, gts

def evaluate(gt_dir, renders_dir):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    renders, gts = readImages(renders_dir, gt_dir)
    
    #renders = renders[1:]

    ssims = []
    psnrs = []
    #lpips_vggs = []
    lpips_alexs = []
    
    n_imgs = min(len(renders), len(gts))
    print("renders: ", len(renders))
    print("gts: ", len(gts))
    print("\nLen: ", n_imgs)

    for idx in tqdm(range(n_imgs), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx])[0])
        psnrs.append(psnr(renders[idx], gts[idx]))
        #lpips_vggs.append(lpips(renders[idx], gts[idx], net_type='vgg'))
        lpips_alexs.append(lpips(renders[idx], gts[idx], net_type='alex'))

    print("Scene: ",  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("Scene: ",  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    #print("Scene: ", scene_dir,  "LPIPS_VGG: {:>12.7f}".format(torch.tensor(lpips_vggs).mean(), ".5"))
    print("Scene: ",  "LPIPS_ALEX: {:>12.7f}".format(torch.tensor(lpips_alexs).mean(), ".5"))
    print("")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--renders_dir', type=str)

    args = parser.parse_args()
    evaluate(args.gt_dir, args.renders_dir)