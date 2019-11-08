from __future__ import division

"""
Script that caches polygons data for future training
"""
"""
Script that caches polygons data for future training
"""
import sys
import os
sys.path.insert(0, os.path.join('..', 'utils'))


import pandas as pd
import extra_functions
from tqdm import tqdm
import h5py
import numpy as np
import tifffile as tiff

#data_path = '../../Data/dstl_data'
data_path = '/home/jxu3/Data/dstl_data'

train_id = ['6060_2_3']
validation_id = ['6110_4_0']
unlabel_id = ['6150_2_3']


def cache_train_16():
    train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))

    print('num_label_train_images =', len(train_id))
    print('num_unlabel_train_images =', len(unlabel_id))
    print('num_validation_images =', len(validation_id))

    min_train_height, min_train_width = 3328, 3328
    image_rows, image_cols = min_train_height, min_train_width

    num_train = len(train_id)
    num_channels = 3
    num_mask_channels = 10

    num_unlabeled_train, num_validation = len(unlabel_id), len(validation_id)

    f_labeled = h5py.File(os.path.join(data_path, 'train_label.h5'), 'w', compression='blosc:lz4', compression_opts=9)
    f_unlabeled = h5py.File(os.path.join(data_path, 'train_unlabel.h5'), 'w', compression='blosc:lz4', compression_opts=9)
    f_validation = h5py.File(os.path.join(data_path, 'validation.h5'), 'w', compression='blosc:lz4', compression_opts=9)

    imgs_unlabeled = f_unlabeled.create_dataset('polygons', (num_unlabeled_train, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_unlabeled_mask = f_unlabeled.create_dataset('train_mask', (num_unlabeled_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    imgs_labeled = f_labeled.create_dataset('polygons', (num_train, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_labeled_mask = f_labeled.create_dataset('train_mask', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    imgs_validation = f_validation.create_dataset('polygons', (num_validation, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_validation_mask = f_validation.create_dataset('train_mask', (num_validation, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    ids, unlabel_ids, validation_ids = [], [], []
    tif_fname = os.path.join(data_path, 'three_band', '{}.tif')

    for i, image_id in enumerate(tqdm(train_id)):
        image = tiff.imread(tif_fname.format(image_id)) / 2047.0
        #image = extra_functions.read_image_16(image_id)
        _, height, width = image.shape
        # populate the following datasets: imgs_labeled, imgs_labeled_mask
        imgs_labeled[i] = image[:, :min_train_height, :min_train_width]
        imgs_labeled_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :min_train_height, :min_train_width]
        ids += [image_id]
    # fix from there: https://github.com/h5py/h5py/issues/441
    f_labeled['train_ids'] = np.array(ids).astype('|S9') # add the 'train_ids' field to f_labeled
    f_labeled.close() # save the data to 'train_label.h5'

    for i, image_id in enumerate(tqdm(unlabel_id)):
        image = tiff.imread(tif_fname.format(image_id)) / 2047.0
        _, height, width = image.shape
        # populate the following datasets: imgs_unlabeled, imgs_unlabeled_mask
        imgs_unlabeled[i] = image[:, :min_train_height, :min_train_width]
        imgs_unlabeled_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :min_train_height, :min_train_width]

        unlabel_ids += [image_id]
    f_unlabeled['train_ids'] = np.array(unlabel_ids).astype('|S9') # add the 'train_ids' field to f_unlabeled
    f_unlabeled.close() # save the data to 'train_label.h5'

    for i, image_id in enumerate(tqdm(validation_id)):
        image = tiff.imread(tif_fname.format(image_id)) / 2047.0
        _, height, width = image.shape
        # populate the following datasets: imgs_validation, imgs_validation_mask
        imgs_validation[i] = image[:, :min_train_height, :min_train_width]
        imgs_validation_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :min_train_height, :min_train_width]
        validation_ids += [image_id]

    f_validation['validation_ids'] = np.array(validation_ids).astype('|S9') # add the 'validation_ids' field
    f_validation.close() # save all data


if __name__ == '__main__':
    cache_train_16()
