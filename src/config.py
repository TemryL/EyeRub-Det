FEATURES = ['accelerometerAccelerationX(G)', 
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

PRETRAINED_MODEL = None

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 2

ENCODER_CFGS = dict(
    learning_rate=1e-3,
    feat_dim=19, 
    max_len=150, 
    d_model=128, 
    num_heads=4,
    num_layers=2, 
    dim_feedforward=4*128, 
    dropout=0.1,
    pos_encoding='learnable', 
    activation='gelu',
    norm='BatchNorm', 
    freeze=False
)

MODEL_CFGS = dict(
    learning_rate=5e-4,
    warmup=100,
    weight_decay=1e-6
)

# Directories
DATA_DIR = "data/"
OUT_DIR = "out/"

# Compute related
ACCELERATOR = "cpu"