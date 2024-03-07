import os
import random
import numpy as np
from keras.models import load_model

from data_organize.utils import get_split, get_images, get_performance, data_loader_tln

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 7


def get_train_val_test():
    # get data from dirs, full path of _n4.nii and labels
    train, val, fuyi, fuer, chenyi = get_split()
    train_set_ri, train_set_nori = get_images(train, train_val='train')
    val_set_ri, val_set_nori = get_images(val, train_val='val')

    random.seed(seed)
    # random_list_train = [train_set_nori[i] for i in random.sample(range(len(train_set_nori)), len(train_set_ri))]
    # random_list_val = [val_set_nori[i] for i in random.sample(range(len(val_set_nori)), len(val_set_ri))]

    train_slice_y = [1] * len(train_set_ri) + [0] * len(train_set_nori)
    val_slice_y = [1] * len(val_set_ri) + [0] * len(val_set_nori)

    train_set_ri.extend(train_set_nori)
    val_set_ri.extend(val_set_nori)

    train_ = list(zip(train_set_ri, train_slice_y))
    random.shuffle(train_)
    train_set_ri, train_slice_y = zip(*train_)

    fuyi_set_ri, fuyi_slice_y, fuer_set_ri, fuer_slice_y, chenyi_set_ri, chenyi_slice_y = [], [], [], [], [], []

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


if __name__ == '__main__':

    x_train, y_train, x_val, y_val, fuyi, y_fuyi, fuer, y_fuer, chenyi, y_chenyi = get_train_val_test()

    train_set, y_train = data_loader_tln(x_train, y_train)
    val_set, y_val = data_loader_tln(x_val, y_val)
    test_fuyi, y_fuyi = data_loader_tln(fuyi, y_fuyi)
    test_fuer, y_fuer = data_loader_tln(fuer, y_fuer)
    test_chenyi, y_chenyi = data_loader_tln(chenyi, y_chenyi)

    model = load_model('./model/T2_20231212_tln_roi_eff.h5')
    pred_fuyi = model.predict(test_fuyi)
    pred_fuer = model.predict(test_fuer)
    pred_chenyi = model.predict(test_chenyi)

    get_performance(pred_fuyi, y_fuyi)
    get_performance(pred_fuer, y_fuer)
    get_performance(pred_chenyi, y_chenyi)

    for i, j,k in zip(fuyi,  y_fuyi, pred_fuyi):
        print(i,j,k)
