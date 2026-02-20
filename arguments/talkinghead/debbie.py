ModelParams = dict(
    loader = "colmapAudio",
    shuffle = False,
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
    
    temporal_embedding_dim = 128,
    aud_embedding_dim = 64,
    gaussian_embedding_dim = 32,
    use_coarse_temporal_embedding = True,
    zero_temporal = False,
    use_anneal = False,
    total_num_frames = None,
    
    emb_posenc_L = 4,
    exp_posenc_L = 0, #0,
)

OptimizationParams = dict(
    dataloader = True,
    batch_size = 1,
    maxtime = None,
    iterations = 50_000,

    lambda_dssim = 0.2,
    num_multiview_ssim = 0,
    use_colmap = True,
    position_lr_max_steps = 50_000,
    deformation_lr_max_steps = 50_000,
    
    #densify_from_iter = 5000,    
    #pruning_from_iter = 600_000,    
    #pruning_interval  = 1000,
    
    opacity_l1_coef_fine = 0.0001,
    densify_until_iter = 50_000,
    
    #deformation_lr_init = 0.00016,
    #deformation_lr_final = 0.000016,

    coef_tv_temporal_embedding = 0, #.0001,
    reg_coef = 1.0,
)