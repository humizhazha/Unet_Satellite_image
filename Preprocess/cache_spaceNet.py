from __future__ import division

"""
Script that caches train data for future training
"""
"""
Script that caches train data for future training
"""
import sys
import os

import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np
import tifffile as tiff
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from shapely.ops import cascaded_union
import numpy as np
import tifffile as tiff



data_path = '../data/'
#Total training image number is 1148
num_label_train_image = 600
num_unlabel_train_image = 300
num_validate_image = 20
image_width = 162
image_height = 162
num_channels = 8
num_mask_channels = 1

train_wkt = pd.read_csv(os.path.join(data_path, 'AOI_3_Paris_Train/summaryData/AOI_3_Paris_Train_Building_Solutions.csv'))


def get_scalers(height, width, x_max, y_min):
    """

    :param height:
    :param width:
    :param x_max:
    :param y_min:
    :return: (xscaler, yscaler)
    """
    w_ = width * (width / (width + 1))
    h_ = height * (height / (height + 1))
    return w_ / x_max, h_ / y_min

def _get_xmax_ymin(image_id):

    return 162, 162

def polygons2mask_layer(height, width, polygons, image_id):
    """

    :param height:
    :param width:
    :param polygons:
    :return:
    """

    x_max, y_min = _get_xmax_ymin(image_id)
    x_scaler, y_scaler = get_scalers(height, width, x_max, y_min)

    polygons = shapely.affinity.scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    img_mask = np.zeros((height, width), np.uint8)

    if not polygons:
        return img_mask

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def generate_mask(image_id, height, width, num_mask_channels=10, train=train_wkt):
    """

    :param image_id:
    :param height:
    :param width:
    :param num_mask_channels: numbers of channels in the desired mask
    :param train: polygons with labels in the polygon format
    :return: mask corresponding to an image_id of the desired height and width with desired number of channels
    """

    mask = np.zeros((height, width, num_mask_channels))

    for mask_channel in range(num_mask_channels):
        poly = train.loc[(train['ImageId'] == image_id), 'PolygonWKT_Pix']
        united_poly = []
        for i in poly:
            polygons = shapely.wkt.loads(i)
            united_poly.append(polygons)
        united_poly = cascaded_union(united_poly)

        mask[ :, :, mask_channel] = polygons2mask_layer(height, width, united_poly, image_id)
    return mask

def cache_train_16():
    image_list = train_wkt['ImageId'].unique()
    label_image_list = image_list[:num_label_train_image]
    unlabel_image_list = image_list[num_label_train_image:num_unlabel_train_image+num_label_train_image]
    validate_image_list = image_list[num_unlabel_train_image+num_label_train_image:num_unlabel_train_image+num_label_train_image+num_validate_image]


    f_label = h5py.File(os.path.join(data_path, 'train_label.h5'), 'w', compression='blosc:lz4', compression_opts=9)
    f_unlabel = h5py.File(os.path.join(data_path, 'train_unlabel.h5'), 'w', compression='blosc:lz4', compression_opts=9)
    f_validation = h5py.File(os.path.join(data_path, 'validation.h5'), 'w', compression='blosc:lz4', compression_opts=9)

    imgs_unlabel = f_unlabel.create_dataset('train', (num_unlabel_train_image, image_width, image_height, num_channels), dtype=np.float16)
    imgs_unlabel_mask = f_unlabel.create_dataset('train_mask', (num_unlabel_train_image, image_width, image_height, num_mask_channels),
                                            dtype=np.uint8)

    imgs = f_label.create_dataset('train', (num_label_train_image,image_width, image_height, num_channels), dtype=np.float16)
    imgs_mask = f_label.create_dataset('train_mask', (num_label_train_image,image_width, image_height, num_mask_channels), dtype=np.uint8)

    imgs_validation = f_validation.create_dataset('train', (num_validate_image, image_width, image_height, num_channels), dtype=np.float16)
    validation_mask = f_validation.create_dataset('train_mask', (num_validate_image, image_width, image_height, num_mask_channels), dtype=np.uint8)

    ids = []
    unlabel_ids=[]
    validation_ids=[]
    tif_fname = os.path.join(data_path, 'AOI_3_Paris_Train/MUL', 'MUL_{}.tif')

    i = 0
    for image_id in tqdm(label_image_list):
   # for image_id in tqdm(['img1912']):
        image = tiff.imread(tif_fname.format(image_id))
        height, width,_ = image.shape
        imgs[i] = image
        imgs_mask[i] = generate_mask('AOI_3_Paris_{}'.format(image_id),height,
                                            width,
                                            num_mask_channels=num_mask_channels,
                                            train=train_wkt)[:image_width, :image_height,:]

        ids += [image_id]
        i += 1

    f_label['train_ids'] = np.array(ids).astype('|S9')
    f_label.close()

    i=0
    for image_id in tqdm(unlabel_image_list):
        image = tiff.imread(tif_fname.format(image_id))
        height, width,_ = image.shape
        imgs_unlabel[i] = image

        imgs_unlabel_mask[i] = generate_mask('AOI_3_Paris_{}'.format(image_id),
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:image_width, :image_height,:]

        unlabel_ids += [image_id]
        i += 1
    f_unlabel['train_ids'] = np.array(unlabel_ids).astype('|S9')
    f_unlabel.close()

    i = 0
    for image_id in tqdm(validate_image_list):
        image = tiff.imread(tif_fname.format(image_id))
        height, width, _ = image.shape
        imgs_validation[i] = image

        validation_mask[i] = generate_mask('AOI_3_Paris_{}'.format(image_id),
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:image_width, :image_height,:]
        validation_ids += [image_id]
        i += 1
    f_validation['validation_ids'] = np.array(validation_ids).astype('|S9')
    f_validation.close()


if __name__ == '__main__':
    cache_train_16()
