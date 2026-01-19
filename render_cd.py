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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from plyfile import PlyData, PlyElement
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

from utils.general_utils import PILtoTorch
from PIL import Image

def get_audio_features(features, att_mode, index):
    if att_mode == 0:
        return features[[index]]
    elif att_mode == 1:
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        if pad_left > 0:
            # pad may be longer than auds, so do not use zeros_like
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
        return auds
    elif att_mode == 2:
        left = index - 4
        right = index + 4
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = features[left:right]
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds
    else:
        raise NotImplementedError(f'wrong att_mode: {att_mode}')


def render_set(model_path, name, iteration, views, gaussians, pipeline, hyperparam=None, aud_path=None, bg_path=None):
    
    render_path = os.path.join(model_path, name, "custom_{}".format(iteration), "renders")
    
    aud_file = np.load(f"{aud_path}")
    aud_file = torch.from_numpy(aud_file)
    aud_file = aud_file.float().permute(0, 2, 1)
    
    makedirs(render_path, exist_ok=True)
    render_images = []
    render_list = []
    deform_vertices = []

    num_down_emb_c = hyperparam.min_embeddings
    num_down_emb_f = hyperparam.min_embeddings    
    
    total_time = 0
    
    real_bg = PILtoTorch(Image.open(bg_path).convert("RGB"), None).cuda()
    
    view_idx = 0
    
    for idx in tqdm(range(len(aud_file))):
        aud_feats = get_audio_features(aud_file, 2, idx).numpy()
        
        view = views[view_idx]
        torso = view.background.cuda()
        background = real_bg * ~torso[3,:,:].to(torch.bool) + torso[:3,:,:]
        
        # change aud
        view.aud = aud_feats
        
        # render
        rendering = render(view, gaussians, pipeline, background, iter=iteration, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)["render"]
        render_with_bg = rendering
        torchvision.utils.save_image(render_with_bg, os.path.join(render_path, view.image_name))
        
        # increase camera counter
        view_idx += 1
        

def render_sets(dataset : ModelParams, hyperparam, opt, iteration : int, pipeline : PipelineParams, aud_path, bg_path):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, duration=None, loader=dataset.loader, opt=opt, load_bg=True)

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, hyperparam=hyperparam, aud_path=aud_path, bg_path=bg_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--aud_path", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--bg_path", type=str, default = None)
    
    # import sys
    # args = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.aud_path, args.bg_path)
    # CUDA_VISIBLE_DEVICES=2 python render.py --model_path output/dynerf/coffee_martini_wo_cam13 --skip_train --configs arguments/dynerf/coffee_martini_wo_cam13.py