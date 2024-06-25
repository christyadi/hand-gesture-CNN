# import tensorflow as tf

import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
# gpus =  tf.config.list_physical_devices('GPU')

# print("Number of GPUs available: ", len(gpus))