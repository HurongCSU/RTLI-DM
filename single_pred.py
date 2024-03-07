import os
import random
from glob import glob

import cv2
import pandas as pd
import tensorflow as tf

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import load_model, Sequential
from keras.regularizers import l2

from data_organize.utils import get_split

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

        # if os.path.exists(img_list[i].replace('t2', 't1')):
        #     t1 = cv2.imread(img_list[i].replace('t2', 't1'))
        #     t1 = cv2.resize(t1, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
        #     t1 = t1 / np.max(t1)
        #     im[:, :, 1] = t1[:, :, 1]

        count = count + 1
        y.append(y_list[i])
        # print(count)
        # plt.imshow(tln + isotropicGrayscaleImage)
        # plt.show()
        img.append(im)
    return np.array(img), np.array(y)


def get_split(excel_file='../data/csv/pp_list.xls'):
    train_df = pd.read_excel(excel_file, sheet_name='xy_train')
    val_df = pd.read_excel(excel_file, sheet_name='xy_val')
    fuyi_df = pd.read_excel(excel_file, sheet_name='fuyi')
    fuer_df = pd.read_excel(excel_file, sheet_name='fuer')
    chenyi_df = pd.read_excel(excel_file, sheet_name='chenyi')
    view_df = pd.read_excel(excel_file, sheet_name='view')

    train = zip(train_df['name'].values, train_df['label'].values)
    val = zip(val_df['name'].values, val_df['label'].values)
    fuyi = zip(fuyi_df['name'].values, fuyi_df['label'].values)
    fuer = zip(fuer_df['name'].values, fuer_df['label'].values)
    chenyi = zip(chenyi_df['name'].values, chenyi_df['label'].values)
    view = view_df['ID'].values

    return train, val, fuyi, fuer, chenyi, view


if __name__ == '__main__':

    model = load_model('./model/T2_20231212_tln_keep_dense.h5')

    train, val, fuyi, fuer, chenyi, view = get_split()
    path = '/home/hurong/PycharmProjects/RI2023/data/RI_slices/keep_dim'

    # features_fuyi = []
    # fuyi_y = []
    # fuyi = zip(['P0001324_20170908'], [0])
    # for name, y_ in fuyi:
    #     tmp = sorted(glob(os.path.join(path, 'fuyi_all', str(name) + '_t2*.png')))
    #     print(len(tmp))
    #     imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #
    #     if len(imgs) == 0:
    #         features_fuyi.append(0)
    #         fuyi_y.append(y_)
    #         continue
    #     import matplotlib.pyplot as plt
    #
    #     # for kk in range(7):
    #     #     plt.imshow(imgs[kk])
    #     #     plt.show()
    #     features = model.predict(imgs)
    #     print(features)
    #     features = features.flatten()
    #     features_fuyi.append(np.max(features))
    #     fuyi_y.append(y_)
    #
    # print(features_fuyi)
    # view =  glob('/home/hurong/3Unet_data/follow_ups/n4' + '/*_n4.nii')
    # view = [os.path.basename(i).split('.nii')[0] for i in view]
    # path = '/home/hurong/3Unet_data/follow_ups/20231213/TLN_keep'
    features_view = []
    print(view)
    for name in view:
        # print(name)

        tmp = sorted(glob(os.path.join(path, 'fuyi_all', str(name) + '*_t2*.png')))
        if len(tmp) == 0:
            tmp = sorted(glob(os.path.join(path, 'fuer_all', str(name) + '*_t2*.png')))
            if len(tmp) == 0:
                tmp = sorted(glob(os.path.join(path, 'chenyi_all', str(name) + '*_t2*.png')))
        # tmp = sorted(glob(os.path.join(path, 'chenyi_all', str(name) + '*_t2*.png')))
        imgs, _ = data_loader_tln(tmp, [0] * len(tmp))

        if len(imgs) == 0:
            features_view.append(0)
            continue

        features = model.predict(imgs)
        features = features.flatten()
        features_view.append(np.max(features))
        print(name, np.max(features))
