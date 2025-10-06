ModelParams = dict(
    loader = "colmapAudio",
    shuffle = True,
    _white_background=False,
    data_device = "cpu",
)

ModelHiddenParams = dict(
    defor_depth = 1,
    net_width = 64,
    no_ds = True,
    no_dr = True,
    no_do = False,
    no_dc = True,
    
    temporal_embedding_dim = 1707, # 256,
    gaussian_embedding_dim = 32,
    use_coarse_temporal_embedding = True,
    zero_temporal = False,
    use_anneal = False,
    total_num_frames = None, #1690, #4270, #4330, #465, #171, 
)

# 465 is n_frames for first 5 vids, 4330 is for all frames, 171 is for vid 17 (single video)
# 4201 is for all vids except vid 003 and 031

OptimizationParams = dict(
    dataloader = True,
    batch_size = 1,
    maxtime = None, #1690, #4270, #4330, #465, #171,
    iterations = 30_000,

    lambda_dssim = 0.2,
    num_multiview_ssim = 0,
    use_colmap = True,
    position_lr_max_steps = 150_000,
    deformation_lr_max_steps = 150_000,
    
    #densify_from_iter = 5000,    
    #pruning_from_iter = 600_000,    
    #pruning_interval  = 1000,
    
    opacity_l1_coef_fine = 0.00001,
    densify_until_iter = 30_000,
    
    #deformation_lr_init = 0.00016,
    #deformation_lr_final = 0.000016,

    coef_tv_temporal_embedding = 0.0001,
    reg_coef = 1.0,
)