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

lr = 5e-4
batch_size = 16
weight_decay = 1e-6
model_name = 'DeepConvLSTM'

# Directories
data_dir = "data/manual/"
train_users = ["user50", "user51", "user53", "user55", "user56", "user59", "user60"]
val_users = ["user52", "user54", "user57", "user58"]
test_users = ["user52", "user54", "user57", "user58"]

# Compute related
accelerator = "gpu"
num_workers = 20
pin_memory = True