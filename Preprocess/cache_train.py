from __future__ import division

"""
Script that caches train data for future training
"""
"""
Script that caches train data for future training
"""

import os
import pandas as pd
import extra_functions
from tqdm import tqdm
import h5py
import numpy as np
import tifffile as tiff

data_path = '../../Data/dstl_data'

gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))
train_id = ['6110_3_1', '6120_2_2', '6140_3_1', '6110_1_2', '6110_4_0']
validation_id = ['6120_2_0']
unlabel_id =['6010_4_0','6010_4_1','6010_4_2','6150_2_3','6170_0_4','6170_4_1']


def cache_train_16():
    train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))

    print('num_label_train_images =', len(train_id))
    print('num_unlabel_train_images =', len(unlabel_id))
    print('num_validation_images =', len(validation_id))

    train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]
    min_train_height = 3328
    min_train_width = 3328



    num_train = train_shapes.shape[0]

    image_rows = min_train_height
    image_cols = min_train_width

    num_channels = 3

    num_mask_channels = 10

    num_unlabeltrain = len(unlabel_id)
    num_validation = len(validation_id)

    f = h5py.File(os.path.join(data_path, 'train_label.h5'), 'w', compression='blosc:lz4', compression_opts=9)
    f_unlabel = h5py.File(os.path.join(data_path, 'train_unlabel.h5'), 'w', compression='blosc:lz4', compression_opts=9)
    f_validation = h5py.File(os.path.join(data_path, 'validation.h5'), 'w', compression='blosc:lz4', compression_opts=9)
    imgs_unlabel = f_unlabel.create_dataset('train', (num_unlabeltrain, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_unlabel_mask = f_unlabel.create_dataset('train_mask', (num_unlabeltrain, num_mask_channels, image_rows, image_cols),
                                            dtype=np.uint8)

    imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_mask = f.create_dataset('train_mask', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    imgs_validation = f_validation.create_dataset('train', (num_validation, num_channels, image_rows, image_cols), dtype=np.float16)
    validation_mask = f_validation.create_dataset('train_mask', (num_validation, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    ids = []
    unlabel_ids=[]
    validation_ids=[]

    i = 0
    for image_id in tqdm(train_id):
        image = tiff.imread("../data/three_band/{}.tif".format(image_id)) / 2047.0
        #image = extra_functions.read_image_16(image_id)
        _, height, width = image.shape
        imgs[i] = image[:, :min_train_height, :min_train_width]
        imgs_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :min_train_height, :min_train_width]

        ids += [image_id]
        i += 1

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')
    f.close()

    i=0
    for image_id in tqdm(unlabel_id):
        image = tiff.imread("../data/three_band/{}.tif".format(image_id)) / 2047.0
        _, height, width = image.shape
        imgs_unlabel[i] = image[:, :min_train_height, :min_train_width]
        imgs_unlabel_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :min_train_height, :min_train_width]

        unlabel_ids += [image_id]
        i += 1
    f_unlabel['train_ids'] = np.array(unlabel_ids).astype('|S9')
    f_unlabel.close()

    i = 0
    for image_id in tqdm(validation_id):
        image = tiff.imread("../data/three_band/{}.tif".format(image_id)) / 2047.0
        _, height, width = image.shape
        imgs_validation[i] = image[:, :min_train_height, :min_train_width]
        validation_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :min_train_height, :min_train_width]
        validation_ids += [image_id]
        i += 1
    f_validation['validation_ids'] = np.array(validation_ids).astype('|S9')
    f_validation.close()


if __name__ == '__main__':
    cache_train_16()
