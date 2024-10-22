"""
Script that caches train data for future training
"""

from __future__ import division
import sys
sys.path.insert(0, '../utils/')
import os
import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np
import tifffile as tiff
import extra_functions

#data_path = '../../Data/dstl_data'
data_path = '../data'

gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))




def cache_test():
    train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))

    print('num_test_images =', train_wkt['ImageId'].nunique())

    train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]

    num_train = train_shapes.shape[0]

    image_rows = 3328
    image_cols = 3328

    num_channels = 3
    num_mask_channels = 10

    f = h5py.File(os.path.join(data_path, 'all.h5'), 'w', compression='blosc:lz4', compression_opts=9)

    imgs = f.create_dataset('image', (train_wkt['ImageId'].nunique(), num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_mask = f.create_dataset('image_mask', (train_wkt['ImageId'].nunique(), num_mask_channels, image_rows, image_cols), dtype=np.float16)

    ids = []

    i = 0
    for image_id in tqdm(sorted(train_wkt['ImageId'].unique())):
        img_fpath = os.path.join(data_path, 'three_band', '{}.tif')
        image = tiff.imread(img_fpath.format(image_id)) / 2047.0
        _, height, width = image.shape
        imgs[i] = image[:, :3328, :3328]
        imgs_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :3328, :3328]

        ids += [image_id]
        i += 1

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')
    f.close()


if __name__ == '__main__':
    cache_test()
