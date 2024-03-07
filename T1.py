import os
import random
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.regularizers import l2

from data_organize.utils import get_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7


def get_images(set_name, institution='xy', train_val=False):
    img_set_ri = []
    img_set_nori = []
    # get instance name
    set_name = [item[0] for item in set_name]
    print(set_name)
    path = '/home/hurong/PycharmProjects/RI2023/data/RI_slices/keep_dim'
    # load all slices
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

    fuyi_set_ri, fuyi_slice_y, fuer_set_ri, fuer_slice_y, chenyi_set_ri, chenyi_slice_y = [], [], [], [], [], []

    return train_set_ri, np.array(train_slice_y), val_set_ri, np.array(val_slice_y), \
        fuyi_set_ri, np.array(fuyi_slice_y), fuer_set_ri, np.array(fuer_slice_y), chenyi_set_ri, np.array(
        chenyi_slice_y)


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

        if os.path.exists(img_list[i].replace('t2', 't1')):
            t1 = cv2.imread(img_list[i].replace('t2', 't1'))
            t1 = cv2.resize(t1, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
            t1 = t1 / np.max(t1)
        else:
            continue
            t1 = np.zeros(im.shape)
        count = count + 1
        y.append(y_list[i])
        img_t2.append(im)
        img_t1.append(t1)
    return np.array(img_t1), np.array(img_t2), np.array(y)


class TrainDCNN:
    def __init__(self, optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=0.00001)):
        self.model = None
        self.optimizer = optimizer
        self.train_gen = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, horizontal_flip=True, vertical_flip=True,
                                            rotation_range=15)
        self.valid_gen = ImageDataGenerator()
        self.batch_size = 16
        self.loss = 'binary_crossentropy'
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
        self.lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        self.save_model = ModelCheckpoint(
            "./model/T2_20231212_tln_keep_dense_t1o_all.h5",
            monitor='val_loss',
            verbose=1,
            mode="min",
            save_best_only=True)

    def gen_flow_for_two_inputs(self, gen, X1, X2, y):
        genX1 = gen.flow(X1, y, batch_size=self.batch_size, seed=666)
        genX2 = gen.flow(X1, X2, batch_size=self.batch_size, seed=666)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            # Assert arrays are equal - this was for peace of mind, but slows down training
            # np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[0]], X1i[1]

    def training(self, _train_set_t1, _train_set_t2, _y_train, _val_t1, _val_t2, _y_val):
        train_data = self.gen_flow_for_two_inputs(self.train_gen, _train_set_t1, _train_set_t2, _y_train)

        valid_data = self.gen_flow_for_two_inputs(self.valid_gen, _val_t1, _val_t2, _y_val)

        train_num = len(_y_train)
        valid_num = len(_y_val)

        import matplotlib.pyplot as plt
        plt.imshow(_train_set_t1[0])
        plt.show()
        plt.imshow(_train_set_t2[0])
        plt.show()

        # self.model1 = keras.applications.densenet.DenseNet121(input_shape=(NET_SIZE, NET_SIZE, 3),
        #                                                       weights='imagenet', include_top=False)

        # for layer in self.model1.layers:
        #     layer.trainable = True
        #     layer._name = layer.name + str("_mirror")
        # model_t2 = load_model('./model/T2_20231212_tln_keep_dense.h5')
        # self.model2 = tf.keras.Model(model_t2.input, model_t2.layers[-11].output)

        # concate = concatenate([self.model1.output, self.model2.output])
        # x = Flatten()(concate)
        # out = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
        # out = Dropout(0.5)(out)
        # out = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(out)
        # out = Dropout(0.5)(out)
        # out = Dense(64, activation="relu", kernel_regularizer=l2(l2=0.01))(out)
        # out = Dropout(0.5)(out)
        # out = Dense(32, activation="relu", kernel_regularizer=l2(l2=0.01))(out)
        # out = Dropout(0.2)(out)
        #
        # outputs = keras.layers.Dense(units=1, activation='sigmoid')(out)

        model = load_model('./model/T2_20231212_tln_keep_dense_t1o_all.h5')
        # model = keras.Model([self.model1.input, self.model2.input], outputs, name='T1')
        # mm.summary()

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['acc', tf.keras.metrics.AUC(name='auc')])
        history = model.fit(x=[_train_set_t1, _train_set_t2], y=_y_train,
                                      steps_per_epoch=train_num / self.batch_size,
                                      validation_data=([_val_t1,_val_t2],_y_val),
                                      validation_steps=valid_num / self.batch_size,
                                      epochs=500,
                                      workers=2,
                                      use_multiprocessing=True,
                                      class_weight={0: 0.2, 1: 0.8},
                                      callbacks=[self.lr_decay, self.save_model, self.early_stop])

        return model


if __name__ == '__main__':
    mod = 'Tra'
    net = 'b4'
    th = 0.5

    x_train, y_train, x_val, y_val, fuyi, y_fuyi, fuer, y_fuer, chenyi, y_chenyi = get_train_val_test()

    train_t1, train_t2, y_train = data_loader_tln(x_train, y_train)
    val_t1, val_t2, y_val = data_loader_tln(x_val, y_val)

    train_T2 = TrainDCNN()
    # train_T2.training(train_t1, train_t2, y_train, val_t1, val_t2, y_val)

    print(len(y_train[y_train == 1]), len(y_train[y_train == 0]))
    print(len(y_val[y_val == 1]), len(y_val[y_val == 0]))

