features = ['accelerometerAccelerationX(G)', 
            'accelerometerAccelerationY(G)',
            'accelerometerAccelerationZ(G)', 
            'motionYaw(rad)', 
            'motionRoll(rad)',
            'motionPitch(rad)', 
            'motionRotationRateX(rad/s)',
            'motionRotationRateY(rad/s)', 
            'motionRotationRateZ(rad/s)',
            'motionUserAccelerationX(G)', 
            'motionUserAccelerationY(G)',
            'motionUserAccelerationZ(G)', 
            'motionQuaternionX(R)',
            'motionQuaternionY(R)', 
            'motionQuaternionZ(R)', 
            'motionQuaternionW(R)',
            'motionGravityX(G)', 
            'motionGravityY(G)', 
            'motionGravityZ(G)'
]

lr = 1e-3
batch_size = 128

unsupervised_dataset = dict(
    features = features, 
    window_size = 150, 
    step_size = 150, 
    normalize = False, 
    mean_mask_length = 3, 
    masking_ratio = 0.15,
    mode = 'separate', 
    distribution = 'geometric', 
    exclude_feats = None
)

encoder_cfgs = dict(
    learning_rate = lr,
    feat_dim = 19, 
    max_len = 150, 
    d_model = 128, 
    num_heads = 16,
    num_layers = 2, 
    dim_feedforward = 512, 
    dropout = 0.1,
    pos_encoding = 'learnable', 
    activation = 'gelu',
    norm = 'BatchNorm', 
    freeze = False,
    warmup = 6000,
    weight_decay = 1e-6,
)

# Directories
data_dir = "data/"
train_path = "data/unsupervised/train.csv"
val_path = "data/unsupervised/val.csv"
out_dir = "out/pretraining_v0/"

# Compute related
accelerator = "gpu"
num_workers = 20
pin_memory = False