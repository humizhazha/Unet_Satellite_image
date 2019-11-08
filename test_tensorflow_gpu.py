'''
 This script tests if the Tensorflow library is using GPU
'''


import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


hello = tf.constant('Hello, TensorFlow!')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))