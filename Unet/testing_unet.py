from __future__ import division
import os
import pickle 
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, '../preprocess/')
sys.path.insert(0, '../lib/')

from operations_2d import *
from utils import *

import h5py



F = tf.app.flags.FLAGS

d_bns = [batch_norm(name='u_bn{}'.format(i,)) for i in range(14)]

# Function to save predicted images as .nii.gz file in results folder
def save_image(direc,i,num):
  img = nib.Nifti1Image(i, None)
  imgname = 'outputimage_unet_'+str(num)+'.nii.gz'
  nib.save(img, os.path.join(direc,imgname))

"""
 Modified 3D U-Net 
"""
def trained_network_dis(patch, reuse=False):
    """
    Parameters:
    * patch - input image for the network
    * reuse - boolean variable to reuse weights
    Returns: 
    * softmax of logits
    """
    with tf.variable_scope('U') as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d_WN(patch, 32, name='u_h0_conv'))
      h1 = lrelu(conv2d_WN(h0, 32, name='u_h1_conv'))
      p1 = avg_pool2D(h1)

      h2 = lrelu(conv2d_WN(p1, 64, name='u_h2_conv'))
      h3 = lrelu(conv2d_WN(h2, 64, name='u_h3_conv'))
      p3 = avg_pool2D(h3)

      h4 = lrelu(conv2d_WN(p3, 128, name='u_h4_conv'))
      h5 = lrelu(conv2d_WN(h4, 128, name='u_h5_conv'))
      p5 = avg_pool2D(h5)

      h6 = lrelu(conv2d_WN(p5, 256, name='u_h6_conv'))
      h7 = lrelu(conv2d_WN(h6, 256, name='u_h7_conv'))

      up1 = deconv2d_WN(h7,256,name='u_up1_deconv')
      up1 = tf.concat([h5,up1],3)
      h8 = lrelu(conv2d_WN(up1, 128, name='u_h8_conv'))
      h9 = lrelu(conv2d_WN(h8, 128, name='u_h9_conv'))
      
      up2 = deconv2d_WN(h9,128,name='u_up2_deconv')
      up2 = tf.concat([h3,up2],3)
      h10 = lrelu(conv2d_WN(up2, 64, name='u_h10_conv'))
      h11 = lrelu(conv2d_WN(h10, 64, name='u_h11_conv'))

      up3 = deconv2d_WN(h11,64,name='u_up3_deconv')
      up3 = tf.concat([h1,up3],3)
      h12 = lrelu(conv2d_WN(up3, 32, name='u_h12_conv'))
      h13 = lrelu(conv2d_WN(h12, 32, name='u_h13_conv'))

      h14 = conv2d_WN(h13, F.num_classes,name='u_h14_conv')

      return tf.nn.softmax(h14)

"""
 Actual 3D U-Net 
"""
def trained_network( patch, phase, pshape, reuse=None):
  """
    Parameters:
    * patch - input image for the network
    * phase - phase for batchnorm
    * pshape - shape of the patch
    * reuse - boolean variable to reuse weights
    Returns: 
    * softmax of logits
    """
  with tf.variable_scope('U') as scope:
    if reuse:
      scope.reuse_variables()

    sh1, sh2, sh3 = int(pshape[0]/4),\
                           int(pshape[0]/2), int(pshape[0])

    h0 = relu(d_bns[0](conv2d(patch, 32, name='u_h0_conv'),phase))
    h1 = relu(d_bns[1](conv2d(h0, 32, name='u_h1_conv'),phase))
    p1 = max_pool2D(h1)

    h2 = relu(d_bns[2](conv2d(p1, 64, name='u_h2_conv'),phase))
    h3 = relu(d_bns[3](conv2d(h2, 64, name='u_h3_conv'),phase))
    p3 = max_pool2D(h3)

    h4 = relu(d_bns[4](conv2d(p3, 128, name='u_h4_conv'),phase))
    h5 = relu(d_bns[5](conv2d(h4, 128, name='u_h5_conv'),phase))
    p5 = max_pool2D(h5)

    h6 = relu(d_bns[6](conv2d(p5, 256, name='u_h6_conv'),phase))
    h7 = relu(d_bns[7](conv2d(h6, 256, name='u_h7_conv'),phase))

    up1 = deconv2d(h7,[F.batch_size,sh1,sh1,256],name='d_up1_deconv')
    up1 = tf.concat([h5,up1],3)
    h8 = relu(d_bns[8](conv2d(up1, 128, name='u_h8_conv'),phase))
    h9 = relu(d_bns[9](conv2d(h8, 128, name='u_h9_conv'),phase))
    
    up2 = deconv2d(h9,[F.batch_size,sh2,sh2,sh2,128],name='d_up2_deconv')
    up2 = tf.concat([h3,up2],3)
    h10 = relu(d_bns[10](conv2d(up2, 64, name='u_h10_conv'),phase))
    h11 = relu(d_bns[11](conv2d(h10, 64, name='u_h11_conv'),phase))

    up3 = deconv2d(h11,[F.batch_size,sh3,sh3,sh3,64],name='d_up3_deconv')
    up3 = tf.concat([h1,up3],3)
    h12 = relu(d_bns[12](conv2d(up3, 32, name='u_h12_conv'),phase))
    h13 = relu(d_bns[13](conv2d(h12, 32, name='u_h13_conv'),phase))

    h14 = conv2d(h13, F.num_classes, name='u_h14_conv')

    return tf.nn.softmax(h14)

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
def get_patches_lab(threeband_vols,label_vols, extraction_step,
                    patch_shape, num_images_training):
    patch_shape_1d = patch_shape[0]
    # Extract patches from input volumes and ground truth
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, 2), dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d), dtype="uint8")
    for idx in range(num_images_training):
        y_length = len(y)
        print(("Extracting Label Patches from Image %2d ....") % (1 + idx))
        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step,
                                        datype="uint8")

        # Select only those who are important for processing
        valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2)) != -1)

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
def preprocess_dynamic_lab(dir,num_classes, extraction_step,patch_shape,num_images_training, type,
                                validating=False,testing=False,num_images_testing=7):
    f = h5py.File(os.path.join("../data", 'test.h5'), 'r')
    test_vols = np.array(f['test'])[:, 2]
    label = np.array(f['test_mask'])[:, type]

    x,y=get_patches_lab(test_vols,label,extraction_step,patch_shape,num_images_training=num_images_training)
    print("Total Extracted Test Patches Shape:",x.shape,y.shape)

    return x, label

"""
 Function to test the model and evaluate the predicted images
 Parameters:
 * patch_shape - shape of the patch
 * extraction_step - stride while extracting patches
"""
def test(patch_shape,extraction_step):
  
  with tf.Graph().as_default():
    test_patches = tf.placeholder(tf.float32, [F.batch_size, patch_shape[0], patch_shape[1], F.num_mod], name='real_patches')
    phase = tf.placeholder(tf.bool)

    # Define the network
    # For using actual 3-D U-Net change ***trained_network*** function both in training and testing
    #output_soft = trained_network(test_patches, phase, patch_shape, reuse=None)
    output_soft = trained_network_dis(test_patches, reuse=None)

    # To convert from one hat form
    output=tf.argmax(output_soft, axis=-1)
    print("Output Patch Shape:",output.get_shape())

    # To load the saved checkpoint
    saver = tf.train.Saver()
    with tf.Session() as sess:
      try:
        load_model(F.best_checkpoint_dir, sess, saver)
        print(" Checkpoint loaded succesfully!....\n")
      except:
        print(" [!] Checkpoint loading failed!....\n")
        return

      # Get patches from test images
      patches_test, labels_test = preprocess_dynamic_lab(F.data_directory,
                                    F.num_classes,extraction_step,patch_shape,
                                    F.number_train_images, F.type_number,validating=F.training,
                                    testing=F.testing,num_images_testing=F.number_test_images)
      total_batches = int(patches_test.shape[0]/F.batch_size)

      # Array to store the prediction results
      predictions_test = np.zeros((patches_test.shape[0],patch_shape[0], patch_shape[1]))

      print("max and min of patches_test:",np.min(patches_test),np.max(patches_test))

      # Batch wise prediction
      print("Total number of Batches: ",total_batches)
      for batch in range(total_batches):
        patches_feed = patches_test[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]
        preds = sess.run(output, feed_dict={test_patches:patches_feed,phase:False})
        predictions_test[batch*F.batch_size:(batch+1)*F.batch_size,:,:]=preds
        print(("Processed_batch:[%8d/%8d]")%(batch,total_batches))

      print("All patches Predicted")

      print("Shape of predictions_test, min and max:",predictions_test.shape,np.min(predictions_test),
                                                                        np.max(predictions_test))

      #To stitch the image back
      images_pred = recompose2D_overlap(predictions_test,3328, 3328, extraction_step[0],
                                                        extraction_step[1])

      print("Shape of Predicted Output Groundtruth Images:",images_pred.shape,
                                                np.min(images_pred), np.max(images_pred),
                                                np.mean(images_pred),np.mean(labels_test))
      
      # To save the images
      for i in range(F.number_test_images):
        pred2d=np.reshape(images_pred[i],(3328*3328))
        lab2d=np.reshape(labels_test[i],(3328*3328))
        #save_image(F.results_dir,images_pred[i],F.number_train_images+i+2)
        F1_score = f1_score(lab2d, pred2d,[0,1,2,3],average=None)

      # Evaluation
      pred2d=np.reshape(images_pred,(images_pred.shape[0]*3328*3328))
      lab2d=np.reshape(labels_test,(labels_test.shape[0]*3328*3328))

      F1_score = f1_score(lab2d, pred2d,[0,1,2,3],average=None)
      print("Testing Dice Coefficient.... ")
      print("Background:",F1_score[0])
      print("CSF:",F1_score[1])
      print("GM:",F1_score[2])
      print("WM:",F1_score[3])

  return






