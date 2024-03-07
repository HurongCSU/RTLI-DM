import os
from glob import glob
import cv2
import numpy as np
from keras.models import load_model
from data_organize.utils import get_split, get_performance

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NET_SIZE = 224


def data_loader_tln(img_list, y_list):
    img_t2 = []
    img_t1 = []
    y = []
    count = 0
    for i in range(len(img_list)):
        # print(i)
        im = cv2.imread(img_list[i])
        if np.max(im) == 0:
            continue

        im = cv2.resize(im, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
        im = im / np.max(im)

        if os.path.exists(img_list[i].replace('t1', 't2')):
            t1 = cv2.imread(img_list[i].replace('t1', 't2'))
            t1 = cv2.resize(t1, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
            t1 = t1 / np.max(t1)
        else:
            continue
        count = count + 1
        y.append(y_list[i])
        img_t2.append(im)
        img_t1.append(t1)
    return np.array(img_t2), np.array(img_t1), np.array(y)



if __name__ == '__main__':

    model = load_model('./model/T2_20231212_tln_keep_dense_t1o_all.h5')

    train, val, fuyi, fuer, chenyi = get_split()
    path = '/home/hurong/PycharmProjects/RI2023/data/RI_slices/keep_dim'

    features_train = []
    train_y = []
    #
    # for name, y_ in train:
    #     tmp = sorted(glob(os.path.join(path, 'xy_all', str(name) + '_t1*.png')))
    #     imgs_t1, imgs_t2, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     if len(imgs_t2) == 0:
    #         print(name)
    #         continue
    #     features = model.predict([imgs_t1, imgs_t2])
    #     features = features.flatten()
    #     features_train.append(np.max(features))
    #     train_y.append(y_)
    #     print(name, np.max(features))

    # features_val = []
    # val_y = []
    # for name, y_ in val:
    #     tmp = sorted(glob(os.path.join(path, 'xy_all', str(name) + '_t1*.png')))
    #     imgs_t1, imgs_t2, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     if len(imgs_t2) == 0:
    #         print(name)
    #         continue
    #     features = model.predict([imgs_t1, imgs_t2])
    #     features = features.flatten()
    #     features_val.append(np.max(features))
    #     val_y.append(y_)
    #     print(name, np.max(features))
    #
    # features_fuyi = []
    # fuyi_y = []
    # for name, y_ in fuyi:
    #     tmp = sorted(glob(os.path.join(path, 'fuyi_all', str(name) + '_t1*.png')))
    #     imgs_t1, imgs_t2, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     if len(imgs_t2) == 0:
    #         print(name)
    #         continue
    #     features = model.predict([imgs_t1, imgs_t2])
    #     features = features.flatten()
    #     features_fuyi.append(np.max(features))
    #     fuyi_y.append(y_)
    #     print(name, np.max(features))
    #
    # features_fuer = []
    # fuer_y = []
    # for name, y_ in fuer:
    #     tmp = sorted(glob(os.path.join(path, 'fuer_all', str(name) + '_t1*.png')))
    #     imgs_t1, imgs_t2, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     if len(imgs_t2) == 0:
    #         print(name)
    #         continue
    #     features = model.predict([imgs_t1, imgs_t2])
    #     features = features.flatten()
    #     features_fuer.append(np.max(features))
    #     fuer_y.append(y_)
    #     print(name, np.max(features))
    #
    features_chenyi = []
    chenyi_y = []
    for name, y_ in chenyi:
        tmp = sorted(glob(os.path.join(path, 'chenyi_all', str(name) + '_t1*.png')))
        imgs_t1, imgs_t2, _ = data_loader_tln(tmp, [y_] * len(tmp))
        if len(imgs_t2) == 0:
            print(name)
            continue
        features = model.predict([imgs_t1, imgs_t2])
        features = features.flatten()
        features_chenyi.append(np.max(features))
        chenyi_y.append(y_)
        print(name, np.max(features))
    #
