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

# Directories
DATA_DIR = "data/"
OUT_DIR = "out/"

# Compute related
ACCELERATOR = "gpu"
NUM_WORKERS = 8
PIN_MEMORY=True