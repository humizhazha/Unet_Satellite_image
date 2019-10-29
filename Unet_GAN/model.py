from __future__ import division
import os
import sys

import h5py
import tensorflow as tf

sys.path.insert(0, os.path.join('..', 'utils'))
sys.path.insert(0, os.path.join('..', 'preprocess'))

from operations_2d import *
#from operations_2d import *

from preprocess import *

import numpy as np
from six.moves import xrange
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


F = tf.app.flags.FLAGS

"""
Model class

"""
class model(object):

  def __init__(self, sess, patch_shape, extraction_step):
    self.sess = sess
    self.patch_shape = patch_shape
    self.extraction_step = extraction_step
    self.g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(4)]
    if F.badGAN:
      self.e_bns = [batch_norm(name='e_bn{}'.format(i,)) for i in range(3)]


  def discriminator(self, patch, reuse=False):
    """
    Parameters:
    * patch - input image for the network
    * reuse - boolean variable to reuse weights
    Returns:
    * logits
    * softmax of logits
    * features extracted from encoding path
    """
    with tf.variable_scope('D') as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d_WN(patch, 32, name='d_h0_conv'))
      h1 = lrelu(conv2d_WN(h0, 32, name='d_h1_conv'))
      p1 = avg_pool2D(h1)

      h2 = lrelu(conv2d_WN(p1, 64, name='d_h2_conv'))
      h3 = lrelu(conv2d_WN(h2, 64, name='d_h3_conv'))
      p3 = avg_pool2D(h3)

      h4 = lrelu(conv2d_WN(p3, 128, name='d_h4_conv'))
      h5 = lrelu(conv2d_WN(h4, 128, name='d_h5_conv'))
      p5 = avg_pool2D(h5)

      h6 = lrelu(conv2d_WN(p5, 256, name='d_h6_conv'))
      h7 = lrelu(conv2d_WN(h6, 256, name='d_h7_conv'))

      up1 = deconv2d_WN(h7, 256, name='d_up1_deconv')
      up1 = tf.concat([h5, up1], 3)
      h8 = lrelu(conv2d_WN(up1, 128, name='d_h8_conv'))
      h9 = lrelu(conv2d_WN(h8, 128, name='d_h9_conv'))

      up2 = deconv2d_WN(h9, 128, name='d_up2_deconv')
      up2 = tf.concat([h3, up2], 3)
      h10 = lrelu(conv2d_WN(up2, 64, name='d_h10_conv'))
      h11 = lrelu(conv2d_WN(h10, 64, name='d_h11_conv'))

      up3 = deconv2d_WN(h11, 64, name='d_up3_deconv')
      up3 = tf.concat([h1, up3], 3)
      h12 = lrelu(conv2d_WN(up3, 32, name='d_h12_conv'))
      h13 = lrelu(conv2d_WN(h12, 32, name='d_h13_conv'))

      h14 = conv2d_WN(h13, F.num_classes, name='d_h14_conv')

      return h14, tf.nn.softmax(h14), h6

  def generator(self, z, phase):
    """
    Parameters:
    * z - Noise vector for generating 3D patches
    * phase - boolean variable to represent phase of operation of batchnorm
    Returns:
    * generated 3D patches
    """
    with tf.variable_scope('G') as scope:
      sh1, sh2, sh3, sh4 = int(self.patch_shape[0] / 16), int(self.patch_shape[0] / 8), \
                           int(self.patch_shape[0] / 4), int(self.patch_shape[0] / 2)

      h0 = linear(z, sh1 * sh1 * 512, 'g_h0_lin')
      h0 = tf.reshape(h0, [F.batch_size, sh1, sh1, 512])
      h0 = relu(self.g_bns[0](h0, phase))

      h1 = relu(self.g_bns[1](deconv2d(h0, [F.batch_size, sh2, sh2, 256],
                                       name='g_h1_deconv'), phase))

      h2 = relu(self.g_bns[2](deconv2d(h1, [F.batch_size, sh3, sh3, 128],
                                       name='g_h2_deconv'), phase))

      h3 = relu(self.g_bns[3](deconv2d(h2, [F.batch_size, sh4, sh4, 64],
                                       name='g_h3_deconv'), phase))

      h4 = deconv2d_WN(h3, F.num_mod, name='g_h4_deconv')

      return tf.nn.tanh(h4)

  def encoder(self, patch, phase):
      """
      Parameters:
      * patch - patches generated from the generator
      * phase - boolean variable to represent phase of operation of batchnorm
      Returns:
      * splitted logits
      """
      with tf.variable_scope('E') as scope:
          h0 = relu(self.e_bns[0](conv2d(patch, 128, 5, 5, 2, 2, name='e_h0_conv'), phase))
          h1 = relu(self.e_bns[1](conv2d(h0, 256, 5, 5, 2, 2, name='e_h1_conv'), phase))
          h2 = relu(self.e_bns[2](conv2d(h1, 512, 5, 5, 2, 2, name='e_h2_conv'), phase))

          h2 = tf.reshape(h2, [h2.shape[0], h2.shape[1] * h2.shape[2] * h2.shape[3]])
          h3 = linear_WN(h2, F.noise_dim * 2, 'e_h3_lin')

          h3 = tf.split(h3, 2, 1)
          return h3


  """
  Defines the Few shot GAN U-Net model and the corresponding losses

  """

  def build_model(self):
    self.patches_lab = tf.placeholder(tf.float32, [F.batch_size, self.patch_shape[0],
                                                   self.patch_shape[1],  F.num_mod],
                                      name='real_images_l')
    self.patches_unlab = tf.placeholder(tf.float32, [F.batch_size, self.patch_shape[0],
                                                     self.patch_shape[1],  F.num_mod],
                                        name='real_images_unl')

    self.z_gen = tf.placeholder(tf.float32, [None, F.noise_dim], name='noise')
    self.labels = tf.placeholder(tf.uint8, [F.batch_size, self.patch_shape[0], self.patch_shape[1],
                                            ], name='image_labels')
    self.phase = tf.placeholder(tf.bool)

    # To make one hot of labels
    self.labels_1hot = tf.one_hot(self.labels, depth=F.num_classes)

    # To generate samples from noise
    self.patches_fake = self.generator(self.z_gen, self.phase)

    # Forward pass through network with different kinds of training patches
    self.D_logits_lab, self.D_probdist, _ = self.discriminator(self.patches_lab, reuse=False)
    self.D_logits_unlab, _, self.features_unlab = self.discriminator(self.patches_unlab, reuse=True)
    self.D_logits_fake, _, self.features_fake = self.discriminator(self.patches_fake, reuse=True)

    # To obtain Validation Output
    self.Val_output = tf.argmax(self.D_probdist, axis=-1)

    # Supervised loss
    # Weighted cross entropy loss (You can play with these values)
    # Weights of different class are: Background- 0.33, CSF- 1.5, GM- 0.83, WM- 1.33
    #class_weights = tf.constant([[0.33, 1.5, 0.83, 1.33]])
    class_weights = tf.constant([[0.33, 1.5]])
    weights = tf.reduce_sum(class_weights * self.labels_1hot, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_logits_lab, labels=self.labels_1hot)
    weighted_losses = unweighted_losses * weights
    self.d_loss_lab = tf.reduce_mean(weighted_losses)

    # Unsupervised loss
    self.unl_lsexp = tf.reduce_logsumexp(self.D_logits_unlab, -1)
    self.fake_lsexp = tf.reduce_logsumexp(self.D_logits_fake, -1)

    # Unlabeled loss
    self.true_loss = - F.tlw * tf.reduce_mean(self.unl_lsexp) + F.tlw * tf.reduce_mean(tf.nn.softplus(self.unl_lsexp))
    # Fake loss
    self.fake_loss = F.flw * tf.reduce_mean(tf.nn.softplus(self.fake_lsexp))
    self.d_loss_unlab = self.true_loss + self.fake_loss

    # Total discriminator loss
    self.d_loss = self.d_loss_lab + self.d_loss_unlab

    # Feature matching loss
    self.g_loss_fm = tf.reduce_mean(
        tf.abs(tf.reduce_mean(self.features_unlab, 0) - tf.reduce_mean(self.features_fake, 0)))

    if F.badGAN:
        # Mean and standard deviation for variational inference loss
        self.mu, self.log_sigma = self.encoder(self.patches_fake, self.phase)
        # Generator Loss via variational inference
        self.vi_loss = gaussian_nll(self.mu, self.log_sigma, self.z_gen)
        # Total Generator Loss
        self.g_loss = self.g_loss_fm + F.vi_weight * self.vi_loss
    else:
        # Total Generator Loss
        self.g_loss = self.g_loss_fm

    t_vars = tf.trainable_variables()

    # define the trainable variables
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    if F.badGAN:
        self.e_vars = [var for var in t_vars if 'e_' in var.name]
    self.saver = tf.train.Saver()

  """
  Train function
  Defines learning rates and optimizers.
  Performs Network update and saves the losses
  """
  def train(self):

    # Instantiate the dataset class
    data = dataset(num_classes=F.num_classes,
                   extraction_step=self.extraction_step,
                   number_images_training=F.number_train_images,
                   batch_size=F.batch_size,
                   patch_shape=self.patch_shape,
                   number_unlab_images_training=F.number_train_unlab_images,
                   data_directory=F.data_directory,
                   type_class = F.type_number)

    # Optimizer operations
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # get the collection of all update operators
    with tf.control_dependencies(update_ops):
        # control_dependencies returns: a context manager that specifies control dependencies for all
        # operations constructed within the context.
        d_optim = tf.train.AdamOptimizer(F.learning_rate_D, beta1=F.beta1D).minimize(self.d_loss, var_list=self.d_vars)
        # d_optim is an Operation that updates the variables in self.d_vars
        g_optim = tf.train.AdamOptimizer(F.learning_rate_G, beta1=F.beta1G).minimize(self.g_loss, var_list=self.g_vars)
        # g_optim is an Operation that updates the variables in self.g_vars
        if F.badGAN:
            e_optim = tf.train.AdamOptimizer(F.learning_rate_E, beta1=F.beta1E).minimize(self.g_loss, var_list=self.e_vars)

    tf.global_variables_initializer().run() # initialize all the variables defined upto this line

    # Load checkpoints if required
    if F.load_chkpt:
        try:
            load_model(F.checkpoint_dir, self.sess, self.saver)
            print("\n [*] Checkpoint loaded succesfully!")
        except:
            print("\n [!] Checkpoint loading failed!")
    else:
        print("\n [*] Checkpoint load not required.")

    # Load the validation data
    patches_val, labels_val_patch, labels_val = \
        preprocess_dynamic_lab(F.data_directory,
                               F.num_classes, self.extraction_step, self.patch_shape,
                               F.number_validate_images,
                               F.type_number,
                               validating=F.training, testing=F.testing)

    predictions_val = np.zeros((patches_val.shape[0], self.patch_shape[0], self.patch_shape[1]), dtype="uint8")
    max_par = 0.0
    max_loss = 100
    for epoch in xrange(int(F.epoch)):
        idx = 0
        batch_iter_train = data.batch_train() # returns labeled batch, unlabeled batch and the labels
        total_val_loss = 0
        total_train_loss_CE = 0
        total_train_loss_UL = 0
        total_train_loss_FK = 0
        total_gen_FMloss = 0

        # go thru all patches
        for patches_lab, patches_unlab, labels in batch_iter_train: # the three items
            # Network update
            sample_z_gen = np.random.uniform(0, 1, [F.batch_size, F.noise_dim]).astype(np.float32)
            # d_optim = tf.train.AdamOptimizer(F.learning_rate_D, beta1=F.beta1D)
            #             .minimize(self.d_loss, var_list=self.d_vars)
            _ = self.sess.run(d_optim,
                              feed_dict={self.patches_lab: patches_lab,
                                         self.patches_unlab: patches_unlab,
                                         self.z_gen: sample_z_gen,
                                         self.labels: labels,
                                         self.phase: True})

            print(self.patches_fake)
            if F.badGAN:
                _, _ = self.sess.run([e_optim, g_optim],
                                     feed_dict={self.patches_unlab: patches_unlab,
                                                self.z_gen: sample_z_gen,
                                                self.z_gen: sample_z_gen,
                                                self.phase: True})
            else:
                _ = self.sess.run(g_optim,
                                  feed_dict={self.patches_unlab: patches_unlab,
                                             self.z_gen: sample_z_gen,
                                             self.z_gen: sample_z_gen,
                                             self.phase: True})

            feed_dict = {self.patches_lab: patches_lab,
                         self.patches_unlab: patches_unlab,
                         self.z_gen: sample_z_gen,
                         self.labels: labels, self.phase: True}

            # Evaluate losses for plotting/printing purposes
            d_loss_lab = self.d_loss_lab.eval(feed_dict)
            d_loss_unlab_true = self.true_loss.eval(feed_dict)
            d_loss_unlab_fake = self.fake_loss.eval(feed_dict)
            g_loss_fm = self.g_loss_fm.eval(feed_dict)

            total_train_loss_CE = total_train_loss_CE + d_loss_lab
            total_train_loss_UL = total_train_loss_UL + d_loss_unlab_true
            total_train_loss_FK = total_train_loss_FK + d_loss_unlab_fake
            total_gen_FMloss = total_gen_FMloss + g_loss_fm

            idx += 1

            if F.badGAN:
                vi_loss = self.vi_loss.eval(feed_dict)
                print((
                          "Epoch:[%2d] [%4d/%4d] Labeled loss:%.2e Unlabeled loss:%.2e Fake loss:%.2e Generator FM loss:%.8f Generator VI loss:%.8f\n") %
                      (epoch, idx, data.num_batches, d_loss_lab, d_loss_unlab_true, d_loss_unlab_fake, g_loss_fm,
                       vi_loss))
            else:
                print((
                          "Epoch:[%2d] [%4d/%4d] Labeled loss:%.2e Unlabeled loss:%.2e Fake loss:%.2e Generator loss:%.8f \n") %
                      (epoch, idx, data.num_batches, d_loss_lab, d_loss_unlab_true, d_loss_unlab_fake, g_loss_fm))

        # save the loss for each epoch
        with open(os.path.join(F.results_dir, 'Train_loss_CE.txt'), 'a') as f:
            f.write('%.2e \n' % total_train_loss_CE)
        with open(os.path.join(F.results_dir, 'Train_loss_UL.txt'), 'a') as f:
            f.write('%.2e \n' % total_train_loss_UL)
        with open(os.path.join(F.results_dir, 'Train_loss_FK.txt'), 'a') as f:
            f.write('%.2e \n' % total_train_loss_FK)
        with open(os.path.join(F.results_dir, 'Train_loss_FM.txt'), 'a') as f:
            f.write('%.2e \n' % total_gen_FMloss)

        # Save the curret model
        save_model(F.checkpoint_dir, self.sess, self.saver)
        if epoch % 10 == 0:
            avg_train_loss_CE = total_train_loss_CE / (idx * 1.0)
            avg_train_loss_UL = total_train_loss_UL / (idx * 1.0)
            avg_train_loss_FK = total_train_loss_FK / (idx * 1.0)
            avg_gen_FMloss = total_gen_FMloss / (idx * 1.0)

            print('\n\n')

            total_batches = int(patches_val.shape[0] / F.batch_size)
            print("Total number of batches for validation: ", total_batches)

    # Prediction of validation patches
            for batch in range(total_batches):
                patches_feed = patches_val[batch * F.batch_size:(batch + 1) * F.batch_size, :, :, :]
                labels_feed = labels_val_patch[batch * F.batch_size:(batch + 1) * F.batch_size, :, :]
                feed_dict = {self.patches_lab: patches_feed,
                         self.labels: labels_feed, self.phase: False}
                preds = self.Val_output.eval(feed_dict)
                val_loss = self.d_loss_lab.eval(feed_dict)

                predictions_val[batch * F.batch_size:(batch + 1) * F.batch_size, :, :] = preds
                print(("Validated Patch:[%8d/%8d]") % (batch, total_batches))
                total_val_loss = total_val_loss + val_loss

    # To compute average patchvise validation loss(cross entropy loss)
            avg_val_loss = total_val_loss / (total_batches * 1.0)

            print("All validation patches Predicted")
            print("Shape of predictions_val, min and max:", predictions_val.shape, np.min(predictions_val),
              np.max(predictions_val))

    # To stitch back the patches into an entire image
            val_image_pred = recompose2D_overlap(predictions_val, 3328, 3328, self.extraction_step[0],
                                             self.extraction_step[1])
            val_image_pred = val_image_pred.astype('uint8')

            print("Shape of Predicted Output Groundtruth Images:", val_image_pred.shape,
              np.unique(val_image_pred),
              np.unique(labels_val),
              np.mean(val_image_pred), np.mean(labels_val))

            pred2d = np.reshape(val_image_pred, (val_image_pred.shape[0] * 3328*3328))
            lab2d = np.reshape(labels_val, (labels_val.shape[0] * 3328*3328))

    # For printing the validation results
            F1_score = f1_score(lab2d, pred2d, [0, 1], average=None)
            print("Validation Dice Coefficient.... ")
            print("Background:", F1_score[0])
            print("Test Class:", F1_score[1])
    # print("GM:", F1_score[2])
    # print("WM:", F1_score[3])

        # To Save the best model
            if (max_par < F1_score[1]):
                max_par =  F1_score[1]
                save_model(F.best_checkpoint_dir, self.sess, self.saver)
                print("Best checkpoint updated from validation results.")

    # To save the losses for plotting
            print("Average Validation Loss:", avg_val_loss)
            with open(os.path.join(F.results_dir, 'Avg_Val_loss_GAN.txt'), 'a') as f:
                f.write('%.2e \n' % avg_val_loss)
            with open(os.path.join(F.results_dir, 'Avg_Train_loss_CE.txt'), 'a') as f:
                f.write('%.2e \n' % avg_train_loss_CE)
            with open(os.path.join(F.results_dir, 'Avg_Train_loss_UL.txt'), 'a') as f:
                f.write('%.2e \n' % avg_train_loss_UL)
            with open(os.path.join(F.results_dir, 'Avg_Train_loss_FK.txt'), 'a') as f:
                f.write('%.2e \n' % avg_train_loss_FK)
            with open(os.path.join(F.results_dir, 'Avg_Train_loss_FM.txt'), 'a') as f:
                f.write('%.2e \n' % avg_gen_FMloss)
    return


"""
To extract patches from a 3D image
"""
def extract_patches(volume, patch_shape, extraction_step,datype='float32'):
  patch_w, patch_d = patch_shape[0], patch_shape[1]
  stride_w, stride_d = extraction_step[0], extraction_step[1]
  img_w, img_d = volume.shape[0],volume.shape[1]

  N_patches_w = (img_w-patch_w)//stride_w+1
  N_patches_d = (img_d-patch_d)//stride_d+1
  N_patches_img =  N_patches_w * N_patches_d
  raw_patch_martrix = np.zeros((N_patches_img,patch_w,patch_d),dtype=datype)
  k=0

  #iterator over all the patches

  for d in range((img_d-patch_d)//stride_d+1):
    for w in range((img_w-patch_w)//stride_w+1):
        raw_patch_martrix[k]=volume[w*stride_w:(w*stride_w)+patch_w, d*stride_d:(d*stride_d)+patch_d]
        k+=1
  assert(k==N_patches_img)
  return raw_patch_martrix

"""
To extract labeled patches from array of 3D labeled images
"""
def get_patches_lab(threeband_vols,
                    label_vols,
                    extraction_step,
                    patch_shape,
                    validating,
                    num_images_training):
    patch_shape_1d = patch_shape[0]
    # Extract patches from input volumes and ground truth
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, 2), dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d), dtype="uint8")
    for idx in range(num_images_training):
        y_length = len(y)
        print(("Extracting Label Patches from Image %2d ....") % (1 + idx))
        label_patches = extract_patches(label_vols[idx],  # the label
                                        patch_shape,
                                        extraction_step,
                                        datype="uint8")

        # Select only those who are important for processing
        if validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2)) != -1)
        else:
            valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2)) > 2000)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d, 2), dtype="float32")))
        y = np.vstack((y, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d), dtype="uint8")))
        #
        y[y_length:, :, :] = label_patches

        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        T1_train = extract_patches(threeband_vols[idx], patch_shape, extraction_step, datype="float32")
        x[y_length:, :, :, 0] = T1_train[valid_idxs]

    return x, y


"""
To preprocess the labeled training data
"""
def preprocess_dynamic_lab(dir,
                           num_classes,
                           extraction_step,
                           patch_shape, num_images_training, type_class,
                           validating=False,
                           testing=False):
    if validating:
        f = h5py.File(os.path.join(F.data_directory, 'validation.h5'), 'r')
    else:
        f = h5py.File(os.path.join(F.data_directory, 'train_label.h5'), 'r')

    label_vols = np.array(f['train'])[:, 2]

    label = np.array(f['train_mask'])[:, type_class]

    x, y = get_patches_lab(label_vols,
                           label,
                           extraction_step,
                           patch_shape,
                           validating,
                           num_images_training=num_images_training)
    print("Total Extracted Labelled Patches Shape:", x.shape, y.shape)
    if testing:
        return x, label
    elif validating:
        return x, y, label
    else:
        return x, y


"""
To extract labeled patches from array of 3D ulabeled images
"""
def get_patches_unlab(unlabel_vols, extraction_step, patch_shape,type_class):
    patch_shape_1d = patch_shape[0]
    # Extract patches from input volumes and ground truth
    #label_ref = np.empty((1, 3345, 3338), dtype="uint8")
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, 2))
    f = h5py.File(os.path.join(F.data_directory, 'train_label.h5'), 'r')
    label_ref = np.array(f['train_mask'])[:, type_class][0]
    for idx in range(len(unlabel_vols)):
        x_length = len(x)
        print(("Extracting Unlabel Patches from Image %2d ....") % (idx+1))
        label_patches = extract_patches(label_ref, patch_shape, extraction_step)

        # Select only those who are important for processing
        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2)) > 2000)

        label_patches = label_patches[valid_idxs]
        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d, 2))))

        unlabel_train = extract_patches(unlabel_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, 0] = unlabel_train[valid_idxs]

    return x


"""
To preprocess the unlabeled training data
"""
def preprocess_dynamic_unlab(dir,extraction_step,patch_shape,num_images_training_unlab, type_class):

    f = h5py.File(os.path.join(F.data_directory, 'train_unlabel.h5'), 'r')
    unlabel_vols  = np.array(f['train'])[:, 2]

    # unlabel_mean = unlabel_vols.mean()
    # unlabel_std = unlabel_vols.std()
    # unlabel_vols = (unlabel_vols - unlabel_mean) / unlabel_std
    # for i in range(unlabel_vols.shape[0]):
    #     unlabel_vols[i] = ((unlabel_vols[i] - np.min(unlabel_vols[i])) /
    #                                     (np.max(unlabel_vols[i])-np.min(unlabel_vols[i])))*255
    #
    # unlabel_vols = unlabel_vols/127.5 -1.
    x=get_patches_unlab(unlabel_vols, extraction_step, patch_shape,type_class)
    print("Total Extracted Unlabelled Patches Shape:",x.shape)
    return x

class dataset(object):
  def __init__(self,num_classes, extraction_step, number_images_training, batch_size,
                    patch_shape, number_unlab_images_training,data_directory,type_class):
    # Extract labelled and unlabelled patches,
    self.batch_size=batch_size

    self.data_lab, self.label = preprocess_dynamic_lab(
                                    data_directory,
                                    num_classes,
                                    extraction_step,
                                    patch_shape,
                                    number_images_training,
                                    type_class)

    self.data_lab, self.label = shuffle(self.data_lab, self.label, random_state=0)

    # note that unlabeled data does not contain any labels
    self.data_unlab = preprocess_dynamic_unlab(data_directory,
                                               extraction_step,
                                               patch_shape,
                                               number_unlab_images_training,
                                               type_class)

    self.data_unlab = shuffle(self.data_unlab, random_state=0)

    # If training, repeat labelled data to make its size equal to unlabelled data
    factor = len(self.data_unlab) // len(self.data_lab)
    print("Factor for labeled images:",factor)
    rem = len(self.data_unlab)%len(self.data_lab)
    temp = self.data_lab[:rem]
    self.data_lab = np.concatenate((np.repeat(self.data_lab, factor, axis=0), temp), axis=0)
    temp = self.label[:rem]
    self.label = np.concatenate((np.repeat(self.label, factor, axis=0), temp), axis=0)
    assert(self.data_lab.shape == self.data_unlab.shape)
    print("Data_shape:",self.data_lab.shape,self.data_unlab.shape)
    print("Data lab max and min:",np.max(self.data_lab),np.min(self.data_lab))
    print("Data unlab max and min:",np.max(self.data_unlab),np.min(self.data_unlab))
    print("Label unique:",np.unique(self.label))

  def batch_train(self):
    print(len(self.data_lab))
    self.num_batches = len(self.data_lab) // self.batch_size
    print(self.num_batches)
    for i in range(self.num_batches):
      yield self.data_lab[i*self.batch_size:(i+1)*self.batch_size],\
             self.data_unlab[i*self.batch_size:(i+1)*self.batch_size],\
                self.label[i*self.batch_size:(i+1)*self.batch_size]


