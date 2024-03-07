import os
import random
from glob import glob
import tensorflow as tf

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import load_model, Sequential
from keras.regularizers import l2

from data_organize.utils import get_split, get_images, get_performance, data_loader_tln

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 7
num_fea = 32
if __name__ == '__main__':

    model = load_model('./model/T2_20231212_tln_roi_eff.h5')
    last_layer_outputs = tf.keras.Model(model.input, model.layers[-3].output)

    train, val, fuyi, fuer, chenyi = get_split()
    path = '/home/hurong/PycharmProjects/RI2023/data/RI_slices/skull_slices'

    features_train = []
    train_y = []
    for name, y_ in train:
        tmp = sorted(glob(os.path.join(path, 'xy_all', str(name) + '_t2*.png')))
        imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
        features_ = np.zeros((30 * num_fea))
        if len(imgs) == 0:
            # features_train.append(features_)
            # train_y.append(y_)
            print(name, y_)
            continue
        features = last_layer_outputs.predict(imgs)
        if y_ == 0:
            for index in range(min(len(imgs), 3)):
                tmp_ = np.zeros((30 * num_fea))
                tmp_fea = features[:index]
                tmp_fea = tmp_fea.flatten()
                tmp_[:len(tmp_fea)] = tmp_fea
                features_train.append(tmp_)
                train_y.append(y_)
        features = features.flatten()
        features_[:len(features)] = features
        features_train.append(features_)
        # print(features.shape)
        # print(len(features_train))
        train_y.append(y_)

    features_val = []
    val_y = []
    for name, y_ in val:
        tmp = sorted(glob(os.path.join(path, 'xy_all', str(name) + '_t2*.png')))
        imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
        features_ = np.zeros((30 * num_fea))
        features = last_layer_outputs.predict(imgs)
        # if y_ == 0:
        #     for index in range(len(imgs)):
        #         tmp_ = np.zeros((30 * num_fea))
        #         tmp_fea = features[:index]
        #         tmp_fea = tmp_fea.flatten()
        #         tmp_[:len(tmp_fea)] = tmp_fea
        #         features_val.append(tmp_)
        #         val_y.append(y_)
        features = features.flatten()
        features_[:len(features)] = features
        features_val.append(features_)
        val_y.append(y_)

    features_train = np.array(features_train)
    train_y = np.array(train_y)
    features_val = np.array(features_val)
    val_y = np.array(val_y)

    print(features_train.shape)
    print(train_y.shape)

    features_fuyi = []
    fuyi_y = []
    for name, y_ in fuyi:
        tmp = sorted(glob(os.path.join(path, 'fuyi_all', str(name) + '_t2*.png')))
        imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
        features_ = np.zeros((30 * num_fea))
        if len(imgs) == 0:
            features_fuyi.append(features_)
            fuyi_y.append(y_)
            continue
        features = last_layer_outputs.predict(imgs)
        features = features.flatten()
        features_[:len(features)] = features
        features_fuyi.append(features_)
        fuyi_y.append(y_)

    # features_fuer = []
    # fuer_y = []
    # for name, y_ in fuer:
    #     tmp = sorted(glob(os.path.join(path, 'fuer_all', str(name) + '_t2*.png')))
    #     imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     features_ = np.zeros((30 * num_fea))
    #     if len(imgs) == 0:
    #         features_fuer.append(features_)
    #         fuer_y.append(y_)
    #         continue
    #     features = last_layer_outputs.predict(imgs)
    #     features = features.flatten()
    #     features_[:len(features)] = features
    #     features_fuer.append(features_)
    #     fuer_y.append(y_)
    #
    # features_chenyi = []
    # chenyi_y = []
    # for name, y_ in chenyi:
    #     tmp = sorted(glob(os.path.join(path, 'chenyi_all', str(name) + '_t2*.png')))
    #     imgs, _ = data_loader_tln(tmp, [y_] * len(tmp))
    #     features_ = np.zeros((30 * num_fea))
    #     if len(imgs) == 0:
    #         features_chenyi.append(features_)
    #         chenyi_y.append(y_)
    #         continue
    #     features = last_layer_outputs.predict(imgs)
    #     features = features.flatten()
    #     features_[:len(features)] = features
    #     features_chenyi.append(features_)
    #     chenyi_y.append(y_)

    fin_model = Sequential()
    fin_model.add(Dense(1024, input_shape=(30 * num_fea,), activation='relu', kernel_regularizer=l2(0.01)))
    fin_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    fin_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    fin_model.add(Dense(1, activation='sigmoid'))
    fin_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    save_model = ModelCheckpoint(
        "model/mlpb4_t2_3_sorted1.h5",
        monitor='val_loss',
        verbose=1,
        mode="auto",
        save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

    features_fuyi = np.array(features_fuyi)
    fuyi_y = np.array(fuyi_y)
    # features_fuer = np.array(features_fuer)
    # fuer_y = np.array(fuer_y)
    # features_chenyi = np.array(features_chenyi)
    # chenyi_y = np.array(chenyi_y)

    fin_model.fit(features_train, train_y, epochs=200, batch_size=64, validation_data=(features_val, val_y),
                  class_weight={0: 0.2, 1: 0.8}, callbacks=[save_model, early_stop, lr_decay])

    fin_model = load_model('model/mlpb4_t2_3_sorted1.h5')

    pred_train = fin_model.predict(features_train)
    pred_val = fin_model.predict(features_val)
    pred_fuyi = fin_model.predict(features_fuyi)
    # pred_fuer = fin_model.predict(features_fuer)
    # pred_chenyi = fin_model.predict(features_chenyi)

    get_performance(pred_train, train_y)
    get_performance(pred_val, val_y)
    get_performance(pred_fuyi, fuyi_y)
    # get_performance(pred_fuer, fuer_y)
    # get_performance(pred_chenyi, chenyi_y)

    train, val, fuyi, fuer, chenyi = get_split()
    for i, j in zip(fuyi, pred_fuyi):
        print(i, j)
