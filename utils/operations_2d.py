import tensorflow as tf
import numpy as np

F = tf.flags.FLAGS


def gaussian_nll(mu, log_sigma, noise):
    NLL = tf.reduce_sum(log_sigma, 1) + \
              tf.reduce_sum(((noise - mu)/(1e-8 + tf.exp(log_sigma)))**2,1)/2.
    return tf.reduce_mean(NLL)


def conv2d(input_, output_dim,k_d=3, k_w=3,
                  s_d=1, s_w=1, stddev=0.05, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_d, k_w, input_.get_shape()[-1], output_dim],
                              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, s_d, s_w, 1], padding='SAME')
    biases = tf.get_variable('biases', [output_dim],
                                    initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv


def deconv2d(input_, output_shape,k_d=2, k_w=2,
                s_d=2, s_w=2, stddev=0.05, name="deconv3d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_d, k_w, output_shape[-1], input_.get_shape()[-1]],
                                    initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                          strides=[1, s_d, s_w, 1], padding="SAME")
    biases = tf.get_variable('biases', [output_shape[-1]],
                                            initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    return deconv


def relu(x, name="relu"):
  return tf.nn.relu(x)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def max_pool2D(input_,k_d=2, k_w=2, s_d=2, s_w=2):
  return tf.contrib.layers.max_pool2d(input_,[k_d, k_w],stride=[s_d, s_w] , padding='SAME')

def avg_pool2D(input_,k_d=2, k_w=2, s_d=2, s_w=2):
  return tf.contrib.layers.avg_pool2d(input_,[k_d, k_w],stride=[s_d,s_w] , padding='SAME')


def linear(input_, output_size, scope=None, stddev=0.05, bias_start=0.0):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
                  initializer=tf.constant_initializer(bias_start))
    return tf.matmul(input_, matrix) + bias



def instance_norm(x,phase=False,name="instance_norm"):
  epsilon = 1e-9
  mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
  return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def int_shape(x):
  return list(map(int, x.get_shape()))

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def conv2d_WN(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', init_scale=1., name="conv_WN", init=False, ema=None, **kwargs):
    ''' convolutional layer '''
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride +[1], pad), b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2,3])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.identity(x)
        return x

def deconv2d_WN(x, num_filters, filter_size=[2,2], stride=[2,2], pad='SAME', init_scale=1., name="deconv_WN", init=False, ema=None, **kwargs):
    ''' transposed convolutional layer '''
    xs = int_shape(x)


    if pad=='SAME':
        target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
    # else:
    #     target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema,
                              shape=filter_size+[num_filters,int(x.get_shape()[-1])],
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema,
                              shape=[num_filters],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema,
                              shape=[num_filters],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 2])

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(x, W, target_shape, [1] + stride +[1], padding=pad)

        x = tf.nn.bias_add(x, b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2,3])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.identity(x)
        return x

def linear_WN(x, num_units, name="linear_WN", init_scale=1., init=False, ema=None, **kwargs):
    ''' fully connected layer '''
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=[int(x.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        if init: # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale/tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g*scale_init), b.assign_add(-m_init*scale_init)]):
                x = tf.identity(x)
        return x

def recompose2D_overlap(preds, img_w, img_d, stride_w, stride_d):
  patch_w = preds.shape[1]
  patch_d = preds.shape[2]
  N_patches_w = (img_w-patch_w)//stride_w+1
  N_patches_d = (img_d-patch_d)//stride_d+1
  N_patches_img = N_patches_w * N_patches_d
  print("N_patches_w: " ,N_patches_w)
  print("N_patches_d: " ,N_patches_d)
  print("N_patches_img: ",N_patches_img)
  assert(preds.shape[0]%N_patches_img==0)
  N_full_imgs = preds.shape[0]//N_patches_img
  print("According to the dimension inserted, there are " \
          +str(N_full_imgs) +" full images (of "+str(img_w)+"x" +str(img_d) +" each)")
  # itialize to zero mega array with sum of Probabilities
  raw_pred_martrix = np.zeros((N_full_imgs,img_w,img_d))
  raw_sum = np.zeros((N_full_imgs,img_w,img_d))
  final_matrix = np.zeros((N_full_imgs,img_w,img_d),dtype='uint16')

  k = 0
  # iterator over all the patches
  for i in range(N_full_imgs):
      for w in range((img_w-patch_w)//stride_w+1):
        for d in range((img_d-patch_d)//stride_d+1):
          raw_pred_martrix[i,w*stride_w:(w*stride_w)+patch_w,d*stride_d:(d*stride_d)+patch_d]+=preds[k]
          raw_sum[i, w*stride_w:(w*stride_w)+patch_w,d*stride_d:(d*stride_d)+patch_d]+=1.0
          k+=1
  assert(k==preds.shape[0])
  #To check for non zero sum matrix
  assert(np.min(raw_sum)>=1.0)
  final_matrix = np.around(raw_pred_martrix/raw_sum)
  return final_matrix