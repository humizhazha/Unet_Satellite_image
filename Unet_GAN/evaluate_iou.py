import pickle
import os
import shapely
import shapely.geometry
import shapely.affinity
import pandas as pd
import tifffile as tiff

import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np


test_ids = '6040_4_4'
##Change the type here. E.g. 5 represents crops
mask_channel = 5


##Change your data path here
data_path = '../data'
train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

##Change your result pickle file here
result_pickle = os.path.join(data_path, 'result_1unlabel.pickle')



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


def generate_mask(image_id, height, width, num_mask_channels=10, train=train_wkt):
    """
    :param image_id:
    :param height:
    :param width:
    :param num_mask_channels: numbers of channels in the desired mask
    :param train: polygons with labels in the polygon format
    :return: mask corresponding to an image_id of the desired height and width with desired number of channels
    """
    mask = np.zeros((num_mask_channels, height, width))
    for mask_channel in range(num_mask_channels):
        poly = train.loc[(train['ImageId'] == image_id)
                         & (train['ClassType'] == mask_channel), 'MultipolygonWKT'].values[0]
        polygons = shapely.wkt.loads(poly)
        mask[mask_channel, :, :] = polygons2mask_layer(height, width, polygons, image_id)
    return mask


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


def polygons2mask(height, width, polygons, image_id):
    num_channels = len(polygons)
    result = np.zeros((num_channels, height, width))
    for mask_channel in range(num_channels):
        result[mask_channel, :, :] = polygons2mask_layer(height, width, polygons[mask_channel], image_id)
    return result


def _get_xmax_ymin(image_id):
    xmax, ymin = gs[gs['ImageId'] == image_id].iloc[0, 1:].astype(float)
    return xmax, ymin



def compute_IOU():

    with open(result_pickle, "rb") as input_file:
        predicted_mask = pickle.load(input_file)

    count_predict = 0
    for i in range(predicted_mask.shape[0]):
        for j in range(predicted_mask.shape[1]):
            if predicted_mask[i][j]==1:
                count_predict=count_predict+1
    print("Predicted count:"+str(count_predict))

    for tid in test_ids:
        real_poly = train_wkt.loc[(train_wkt['ImageId'] == test_ids)
                             & (train_wkt['ClassType'] == mask_channel), 'MultipolygonWKT'].values[0]
        polygons = shapely.wkt.loads(real_poly)
    mask = np.zeros((3328, 3328))
    mask[:, :] = polygons2mask_layer(3328, 3328, polygons, test_ids)

    count_real = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]==1:
                count_real=count_real+1
    print("Real count:"+str(count_real))

    count_common=0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]==1 and predicted_mask[i][j]==1:
                count_common=count_common+1

    iou = count_common/(count_real+count_predict-count_common)
    print("IOU:"+str(iou))

def compute_IOU_on_Validation(predicted_mask,label):

    count_predict = 0
    for i in range(predicted_mask.shape[0]):
        for j in range(predicted_mask.shape[1]):
            if predicted_mask[i][j]==1:
                count_predict=count_predict+1
    print("Predicted count:"+str(count_predict))

    count_real = 0
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j]==1:
                count_real=count_real+1
    print("Real count:"+str(count_real))

    count_common=0
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j]==1 and predicted_mask[i][j]==1:
                count_common=count_common+1

    iou = count_common/(count_real+count_predict-count_common)
    return iou

if __name__ == '__main__':
    compute_IOU()