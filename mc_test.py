import os
import random
from glob import glob

import cv2
import tensorflow as tf

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.regularizers import l2

from data_organize.utils import get_split, get_images, get_performance, data_loader_tln

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7
num_fea = 32

NET_SIZE = 224


def data_loader_tln(img_list, y_list):
    img = []
    y = []
    count = 0
    for i in range(len(img_list)):
        # print(i)
        im = cv2.imread(img_list[i])
        if np.max(im) == 0:
            continue
        im = cv2.resize(im, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
        im = im / np.max(im)
        count = count + 1
        y.append(y_list[i])
        # print(count)
        # plt.imshow(tln + isotropicGrayscaleImage)
        # plt.show()
        img.append(im)
    return np.array(img), np.array(y)

if __name__ == '__main__':

    model = load_model('./model/T2_20231212_tln_keep_dense_mc.h5')

    train, val, fuyi, fuer, chenyi = get_split()
    path = '/home/hurong/PycharmProjects/RI2023/data/RI_slices/keep_dim'


    train_y = []
    mc =[]
    for i in range(10):
        features_train = []
        names = []
        for name, y_ in train:
            tmp = sorted(glob(os.path.join(path, 'xy_all', str(name) + '*.png')))
            names.extend(tmp)
            imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
            if len(imgs) == 0:
                continue

            features = model.predict(imgs)
            features = features.flatten()
        features_train.extend(features)
        mc.append(features_train)

