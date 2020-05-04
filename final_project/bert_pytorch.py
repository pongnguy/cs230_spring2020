"""
File to do a baseline BERT model fine-tuning using PyTorch

See this article https://mccormickml.com/2019/07/22/BERT-fine-tuning/


A. Wechselberger, 2020
"""

import tensorflow as tf

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')