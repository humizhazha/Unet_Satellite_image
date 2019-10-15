import numpy as np
from six.moves import xrange
from sklearn.metrics import f1_score
import tifffile as tiff
import pandas as pd
import os
import pickle
import h5py
import tensorflow as tf
from sklearn.utils import shuffle

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
                    patch_shape,validating, num_images_training):
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
        if validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2)) != -1)
        else:
            valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2)) > 1000)

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
def preprocess_dynamic_lab(dir,num_classes, extraction_step,patch_shape,num_images_training, type_class, validating=False,
                           testing=False):
    if validating:
        f = h5py.File(os.path.join("../data", 'validation.h5'), 'r')
    else:
        f = h5py.File(os.path.join("../data", 'train_label.h5'), 'r')

    label_vols = np.array(f['train'])[:, 2]

    label = np.array(f['train_mask'])[:, type_class]

    x,y=get_patches_lab(label_vols,label,extraction_step,patch_shape,validating, num_images_training=num_images_training)
    print("Total Extracted Labelled Patches Shape:",x.shape,y.shape)
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
    f = h5py.File(os.path.join("../data", 'train_label.h5'), 'r')
    label_ref = np.array(f['train_mask'])[:, type_class][0]
    for idx in range(len(unlabel_vols)):
        x_length = len(x)
        print(("Extracting Unlabel Patches from Image %2d ....") % (idx+1))
        label_patches = extract_patches(label_ref, patch_shape, extraction_step)

        # Select only those who are important for processing
        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2)) > 1000)

        label_patches = label_patches[valid_idxs]
        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d, 2))))

        unlabel_train = extract_patches(unlabel_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, 0] = unlabel_train[valid_idxs]

    return x


"""
To preprocess the unlabeled training data
"""
def preprocess_dynamic_unlab(dir,extraction_step,patch_shape,num_images_training_unlab, type_class):

    f = h5py.File(os.path.join("../data", 'train_unlabel.h5'), 'r')
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
                                data_directory,num_classes,extraction_step,
                                        patch_shape,number_images_training,type_class)

    self.data_lab, self.label = shuffle(self.data_lab, self.label, random_state=0)
    self.data_unlab = preprocess_dynamic_unlab(data_directory,extraction_step,
                                                patch_shape, number_unlab_images_training,type_class)
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
    # filename = "label.pickle"
    # filehandler = open(filename,'wb')
    # pickle.dump(self.label,filehandler)
    # filehandler.close()
    print("Label unique:",np.unique(self.label))

  def batch_train(self):
    self.num_batches = len(self.data_lab) // self.batch_size
    for i in range(self.num_batches):
      yield self.data_lab[i*self.batch_size:(i+1)*self.batch_size],\
             self.data_unlab[i*self.batch_size:(i+1)*self.batch_size],\
                self.label[i*self.batch_size:(i+1)*self.batch_size]

