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
normalize = False

lr = 1e-3
batch_size = 8
weight_decay = 1e-6
model_name = 'CNN'

# Directories
data_dir = "data/"
train_path = "data/supervised/train"
val_path = "data/supervised/val"
out_dir = "logs/CNN/"

# Compute related
accelerator = "gpu"
num_workers = 20
pin_memory = False