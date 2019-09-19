# import numpy as np
# import h5py
# import os
# f = h5py.File(os.path.join("../data", 'train_label.h5'), 'r')
#
# X_train = np.array(f['train'])[:, 2]
#
# y_train = np.array(f['train_mask'])[:, 3]
# label_mean = X_train.mean()
# label_std = X_train.std()
# label_vols = (X_train - label_mean) / label_std
# print(y_train.shape)
# print(X_train.shape)
# print(np.max(label_vols))
# print(np.max(y_train))

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
