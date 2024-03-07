import os
import random
from glob import glob

import cv2
import numpy as np
from keras.models import load_model
from data_organize.utils import get_split, get_performance, bootstrap_auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7


def get_images(set_name, institution='xy', train_val=False):
    img_set_ri = []
    img_set_nori = []
    set_name = [item[0] for item in set_name]
    print(set_name)
    path = '/home/hurong/PycharmProjects/RI2023/data/RI_slices/keep_dim'
    if train_val:
        institution = os.path.join(institution, train_val)
    for i in set_name:
        img_set_ri.extend(glob(os.path.join(path, institution, 'RE', str(i) + '_t2*.png')))
        img_set_nori.extend(glob(os.path.join(path, institution, 'NRE', str(i) + '_t2*.png')))
    return img_set_ri, img_set_nori


def get_train_val_test():
    # get data from dirs, full path of _n4.nii and labels
    train, val, fuyi, fuer, chenyi = get_split()
    train_set_ri, train_set_nori = get_images(train, train_val='train')
    val_set_ri, val_set_nori = get_images(val, train_val='val')

    random.seed(seed)
    train_slice_y = [1] * len(train_set_ri) + [0] * len(train_set_nori)
    val_slice_y = [1] * len(val_set_ri) + [0] * len(val_set_nori)

    train_set_ri.extend(train_set_nori)
    val_set_ri.extend(val_set_nori)

    train_ = list(zip(train_set_ri, train_slice_y))
    random.shuffle(train_)
    train_set_ri, train_slice_y = zip(*train_)

    fuyi_set_ri, fuyi_set_nori = get_images(fuyi, 'fuyi')
    fuer_set_ri, fuer_set_nori = get_images(fuer, 'fuer')
    chenyi_set_ri, chenyi_set_nori = get_images(chenyi, 'chenyi')

    fuyi_slice_y = [1] * len(fuyi_set_ri) + [0] * len(fuyi_set_nori)
    fuer_slice_y = [1] * len(fuer_set_ri) + [0] * len(fuer_set_nori)
    chenyi_slice_y = [1] * len(chenyi_set_ri) + [0] * len(chenyi_set_nori)
    fuyi_set_ri.extend(fuyi_set_nori)
    fuer_set_ri.extend(fuer_set_nori)
    chenyi_set_ri.extend(chenyi_set_nori)

    return train_set_ri, np.array(train_slice_y), val_set_ri, np.array(val_slice_y), \
        fuyi_set_ri, np.array(fuyi_slice_y), fuer_set_ri, np.array(fuer_slice_y), chenyi_set_ri, np.array(
        chenyi_slice_y)


NET_SIZE = 224


def data_loader_tln(img_list, y_list):
    img = []
    y = []
    count = 0
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
    mod = 'Tra'
    net = 'b4'
    th = 0.5

    x_train, y_train, x_val, y_val, fuyi, y_fuyi, fuer, y_fuer, chenyi, y_chenyi = get_train_val_test()

    train_set, y_train = data_loader_tln(x_train, y_train)
    val_set, y_val = data_loader_tln(x_val, y_val)
    test_fuyi, y_fuyi = data_loader_tln(fuyi, y_fuyi)
    test_fuer, y_fuer = data_loader_tln(fuer, y_fuer)
    test_chenyi, y_chenyi = data_loader_tln(chenyi, y_chenyi)

    print(len(y_train[y_train == 1]), len(y_train[y_train == 0]))
    print(len(y_val[y_val == 1]), len(y_val[y_val == 0]))
    print(len(y_fuyi[y_fuyi == 1]), len(y_fuyi[y_fuyi == 0]))
    print(len(y_fuer[y_fuer == 1]), len(y_fuer[y_fuer == 0]))
    print(len(y_chenyi[y_chenyi == 1]), len(y_chenyi[y_chenyi == 0]))

    model = load_model('./model/T2_20231212_tln_keep_dense.h5')

    train_pred = model.predict(train_set)
    val_pred = model.predict(val_set)
    fuyi_pred = model.predict(test_fuyi)
    fuer_pred = model.predict(test_fuer)
    chenyi_pred = model.predict(test_chenyi)

    th = 0.37

    get_performance(train_pred, y_train, th)
    get_performance(val_pred, y_val, th)
    get_performance(fuyi_pred, y_fuyi, th)
    get_performance(fuer_pred, y_fuer, th)
    get_performance(chenyi_pred, y_chenyi, th)

    statistics = bootstrap_auc(y_train, train_pred, [0, 1])
    print("均值:", np.mean(statistics, axis=1))
    print("最大值:", np.max(statistics, axis=1))
    print("最小值:", np.min(statistics, axis=1))
    statistics = bootstrap_auc(y_val, val_pred, [0, 1])
    print("均值:", np.mean(statistics, axis=1))
    print("最大值:", np.max(statistics, axis=1))
    print("最小值:", np.min(statistics, axis=1))
    statistics = bootstrap_auc(y_fuyi, fuyi_pred, [0, 1])
    print("均值:", np.mean(statistics, axis=1))
    print("最大值:", np.max(statistics, axis=1))
    print("最小值:", np.min(statistics, axis=1))
    statistics = bootstrap_auc(y_fuer, fuer_pred, [0, 1])
    print("均值:", np.mean(statistics, axis=1))
    print("最大值:", np.max(statistics, axis=1))
    print("最小值:", np.min(statistics, axis=1))
    statistics = bootstrap_auc(y_chenyi, chenyi_pred, [0, 1])
    print("均值:", np.mean(statistics, axis=1))
    print("最大值:", np.max(statistics, axis=1))
    print("最小值:", np.min(statistics, axis=1))


