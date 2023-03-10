DEPTH_LAYERS = 50
POSE_LAYERS = 18
FRAME_IDS = [0, 1, -1]
IMGS_PER_GPU = 12
HEIGHT = 320
WIDTH = 1024


data = dict(
    name = 'kitti_odom',
    split = 'odom',
    height = HEIGHT,
    width = WIDTH,
    frame_ids = FRAME_IDS,
    in_path = '../kitti_odometry',
    gt_depth_path = './pre_trained/gt_depths.npz',
    png = True,
    stereo_scale = True if 's' in FRAME_IDS else False,
    CZDA = True,
)

model = dict(
    name = 'mono_hrnet',
    depth_num_layers = DEPTH_LAYERS,
    pose_num_layers = POSE_LAYERS,
    frame_ids = FRAME_IDS,
    imgs_per_gpu = IMGS_PER_GPU,
    height = HEIGHT,
    width = WIDTH,
    scales = [0],
    min_depth = 0.1,
    max_depth = 100.0,
    depth_pretrained_path = './pre_trained/hrnetv2_w18_imagenet_pretrained.pth',
    pose_pretrained_path =  './pre_trained/resnet18-5c106cde.pth',
    automask = False if 's' in FRAME_IDS else True,
    disp_norm = False if 's' in FRAME_IDS else True,
    smoothness_weight=1e-3,
)

resume_from = None
finetune = None
total_epochs = 40
imgs_per_gpu = IMGS_PER_GPU
learning_rate = 1e-4
workers_per_gpu = 4
validate = False

optimizer = dict(type='Adam', lr=learning_rate, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[25, 30],
    gamma=0.5,
)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50,
                  hooks=[dict(type='TextLoggerHook'),])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]
