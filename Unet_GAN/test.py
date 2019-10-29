from __future__ import division
import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import h5py
import sys
import pickle

sys.path.insert(0, os.path.join('..', 'utils'))
from operations_2d import *
#from operations_2d import *

F = tf.app.flags.FLAGS


# Function to save predicted images as .nii.gz file in results folder
def save_image(direc, i, num):
    file_name = 'result'+str(num)+'.pickle'
    filehandler = open(file_name,'wb')
    pickle.dump(i,filehandler)
    filehandler.close()


# Same discriminator network as in model file
def trained_dis_network(patch, reuse=False):
    """
    Parameters:
    * patch - input image for the network
    * reuse - boolean variable to reuse weights
    Returns:
    * softmax of logits
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


def test(patch_shape, extraction_step):
    with tf.Graph().as_default():
        test_patches = tf.placeholder(tf.float32, [F.batch_size, patch_shape[0], patch_shape[1], F.num_mod], name='real_patches')

        # Define the network
        output_soft = trained_dis_network(test_patches, reuse=None)

        # To convert from one hat form
        output = tf.argmax(output_soft, axis=-1)
        print("Output Patch Shape:", output.get_shape())

        # To load the saved checkpoint
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                load_model("result1/", sess, saver)
                print(" Checkpoint loaded succesfully!....\n")
            except:
                print(" [!] Checkpoint loading failed!....\n")
                return

            # Get patches from test images
            patches_test, labels_test = preprocess_dynamic_lab(F.data_directory,
                                                               F.num_classes, extraction_step, patch_shape,
                                                               F.number_train_images, F.type_number,validating=F.training,
                                                               testing=F.testing,
                                                               num_images_testing=F.number_test_images)
            total_batches = int(patches_test.shape[0] / F.batch_size)

            # Array to store the prediction results
            predictions_test = np.zeros((patches_test.shape[0], patch_shape[0], patch_shape[1]))

            print("max and min of patches_test:", np.min(patches_test), np.max(patches_test))

            print("Total number of Batches: ", total_batches)

            # Batch wise prediction
            for batch in range(total_batches):
                patches_feed = patches_test[batch * F.batch_size:(batch + 1) * F.batch_size, :, :, :]
                preds = sess.run(output, feed_dict={test_patches: patches_feed})
                predictions_test[batch * F.batch_size:(batch + 1) * F.batch_size, :, :] = preds
                print(("Processed_batch:[%8d/%8d]") % (batch, total_batches))

            print("All patches Predicted")

            print("Shape of predictions_test, min and max:", predictions_test.shape, np.min(predictions_test),
                  np.max(predictions_test))

            # To stitch the image back
            images_pred = recompose2D_overlap(predictions_test, 3328,3328, extraction_step[0],extraction_step[1])

            print("Shape of Predicted Output Groundtruth Images:", images_pred.shape,
                  np.min(images_pred), np.max(images_pred),
                  np.mean(images_pred), np.mean(labels_test))

            # To save the images
            for i in range(F.number_test_images):
                pred2d = np.reshape(images_pred[i], (3328*3328))
                lab2d = np.reshape(labels_test[i], (3328*3328))
                save_image(F.results_dir, images_pred[i], F.number_train_images + i + 2)

            # Evaluation
            pred2d = np.reshape(images_pred, (images_pred.shape[0] * 3328*3328))
            lab2d = np.reshape(labels_test, (labels_test.shape[0] * 3328*3328))

            F1_score = f1_score(lab2d, pred2d, [0, 1], average=None)
            print("Testing Dice Coefficient.... ")
            print("Background:", F1_score[0])
            print("CSF:", F1_score[1])
            

    return







