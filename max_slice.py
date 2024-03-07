import os
from glob import glob
import cv2
import numpy as np
from keras.models import load_model
from data_organize.utils import get_split, get_performance

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NET_SIZE = 224


def data_loader_tln(img_list, y_list):
    img = []
    y = []
    for i in range(len(img_list)):
        im = cv2.imread(img_list[i])
        if np.max(im) == 0:
            continue
        im = cv2.resize(im, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
        im = im / np.max(im)
        y.append(y_list[i])
        img.append(im)
    return np.array(img), np.array(y)


if __name__ == '__main__':

    model = load_model('./model/T2_20231212_tln_keep_dense.h5')

    train, val, fuyi, fuer, chenyi = get_split()
    path = '/home/hurong/PycharmProjects/RI2023/data/RI_slices/keep_dim'
    # path = '/home/hurong/3Unet_data/T2_preprocessed/manual'

    # features_train = []
    # train_y = []
    # for name, y_ in train:
    #     tmp = sorted(glob(os.path.join(path, 'xy_all', str(name) + '_t2*.png')))
    #     imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     if len(imgs) == 0:
    #         features_train.append(0)
    #         train_y.append(y_)
    #         continue
    #     features = model.predict(imgs)
    #     features = features.flatten()
    #     features_train.append(np.max(features))
    #     train_y.append(y_)
    #
    # features_val = []
    # val_y = []
    # for name, y_ in val:
    #     tmp = sorted(glob(os.path.join(path, 'xy_all', str(name) + '_t2*.png')))
    #     imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     features = model.predict(imgs)
    #     features = features.flatten()
    #     features_val.append(np.max(features))
    #     val_y.append(y_)
    #
    # features_train = np.array(features_train)
    # train_y = np.array(train_y)
    # features_val = np.array(features_val)
    # val_y = np.array(val_y)

    features_fuyi = []
    fuyi_y = []
    for name, y_ in fuyi:
        tmp = sorted(glob(os.path.join(path, 'fuyi', str(name) + '_t2*.png')))
        imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
        if len(imgs) == 0:
            features_fuyi.append(0)
            fuyi_y.append(y_)
            continue
        features = model.predict(imgs)
        features = features.flatten()
        features_fuyi.append(np.max(features))
        fuyi_y.append(y_)

    # features_fuer = []
    # fuer_y = []
    # for name, y_ in fuer:
    #     tmp = sorted(glob(os.path.join(path, 'fuer', str(name) + '_t2*.png')))
    #     imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     if len(imgs) == 0:
    #         features_fuer.append(0)
    #         fuer_y.append(y_)
    #         continue
    #     features = model.predict(imgs)
    #     features = features.flatten()
    #     features_fuer.append(np.max(features))
    #     fuer_y.append(y_)

    # features_chenyi = []
    # chenyi_y = []
    # for name, y_ in chenyi:
    #     tmp = sorted(glob(os.path.join(path, 'chenyi', str(name) + '_t2*.png')))
    #     imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     if len(imgs) == 0:
    #         features_chenyi.append(0)
    #         chenyi_y.append(y_)
    #         continue
    #     features = model.predict(imgs)
    #     features = features.flatten()
    #     features_chenyi.append(np.max(features))
    #     chenyi_y.append(y_)

    features_fuyi = np.array(features_fuyi)
    fuyi_y = np.array(fuyi_y)
    # features_fuer = np.array(features_fuer)
    # fuer_y = np.array(fuer_y)
    # features_chenyi = np.array(features_chenyi)
    # chenyi_y = np.array(chenyi_y)
    #
    # get_performance(features_train, train_y, 0.83)
    # get_performance(features_val, val_y, 0.83)
    get_performance(features_fuyi, fuyi_y, 0.83)
    # get_performance(features_fuer, fuer_y, 0.83)
    # get_performance(features_chenyi, chenyi_y, 0.83)

    train, val, fuyi, fuer, chenyi = get_split()

    # for i, j in zip(train, features_train):
    #     print(i, j)
    #
    # for i, j in zip(val, features_val):
    #     print(i, j)
    #
    for i, j in zip(fuyi, features_fuyi):
        print(i, j)
    #
    # for i, j in zip(fuer, features_fuer):
    #     print(i, j)

    # for i, j in zip(chenyi, features_chenyi):
    #     print(i, j)
