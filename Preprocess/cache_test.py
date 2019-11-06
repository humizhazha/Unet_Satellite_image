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

#data_path = '../../data'
data_path = '/home/jxu3/Data/dstl_data'

test_id = ['6120_2_0']


def cache_test():

    train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
    test_wkt = train_wkt[train_wkt['ImageId'].isin(test_id)]
    print('num_test_images =', test_wkt['ImageId'].nunique())

    image_rows, image_cols = 3328, 3328

    num_test = len(test_id)
    num_channels = 3
    num_mask_channels = 10

    f = h5py.File(os.path.join(data_path, 'test.h5'), 'w', compression='blosc:lz4', compression_opts=9)
    imgs = f.create_dataset('test', (num_test, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_mask = f.create_dataset('test_mask', (num_test, num_mask_channels, image_rows, image_cols), dtype=np.float16)

    ids = []
    for i, image_id in enumerate(tqdm(sorted(test_id))):
        img_fpath = os.path.join(data_path, 'three_band', '{}.tif')
        image = tiff.imread(img_fpath.format(image_id)) / 2047.0
        _, height, width = image.shape
        imgs[i] = image[:, :3328, :3328]
        imgs_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=test_wkt)[:, :3328, :3328]
        ids += [image_id]

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')
    f.close()


if __name__ == '__main__':
    cache_test()
