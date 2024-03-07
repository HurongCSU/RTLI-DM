import os
import random
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.regularizers import l2

from data_organize.utils import get_split, data_loader_tln, NET_SIZE

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
        img_set_ri.extend(glob(os.path.join(path, institution, 'RE', str(i) + '*.png')))
        img_set_nori.extend(glob(os.path.join(path, institution, 'NRE', str(i) + '*.png')))
    return img_set_ri, img_set_nori


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

    # fuyi_set_ri, fuyi_set_nori = get_images(fuyi, 'fuyi')
    # fuer_set_ri, fuer_set_nori = get_images(fuer, 'fuer')
    # chenyi_set_ri, chenyi_set_nori = get_images(chenyi, 'chenyi')
    # fuyi_slice_y = [1] * len(fuyi_set_ri) + [0] * len(fuyi_set_nori)
    # fuer_slice_y = [1] * len(fuer_set_ri) + [0] * len(fuer_set_nori)
    # chenyi_slice_y = [1] * len(chenyi_set_ri) + [0] * len(chenyi_set_nori)
    # fuyi_set_ri.extend(fuyi_set_nori)
    # fuer_set_ri.extend(fuer_set_nori)
    # chenyi_set_ri.extend(chenyi_set_nori)

    return train_set_ri, np.array(train_slice_y), val_set_ri, np.array(val_slice_y), \
        fuyi_set_ri, np.array(fuyi_slice_y), fuer_set_ri, np.array(fuer_slice_y), chenyi_set_ri, np.array(
        chenyi_slice_y)


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
            "./model/T2_20231212_tln_keep_eff.h5",
            monitor='val_loss',
            verbose=1,
            mode="min",
            save_best_only=True)

    def training(self, _train_set, _y_train, _val_set, _y_val):
        train_data = self.train_gen.flow(_train_set, _y_train, shuffle=True, batch_size=self.batch_size)
        valid_data = self.valid_gen.flow(_val_set, _y_val, shuffle=False, batch_size=self.batch_size)

        train_num = len(_y_train)
        valid_num = len(_y_val)

        self.model = keras.applications.efficientnet.EfficientNetB4(input_shape=(NET_SIZE, NET_SIZE, 3),
                                                                    weights='imagenet', include_top=False)

        # from swintransformer import SwinTransformer
        #
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(224, 224, 3)),
        #     SwinTransformer('swin_tiny_224', include_top=False, pretrained=True),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])

        # x = keras.layers.GlobalAveragePooling2D()(self.model.output)
        x = Flatten()(self.model.output)
        out = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
        out = Dropout(0.5)(out)
        out = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(out)
        out = Dropout(0.5)(out)
        out = Dense(64, activation="relu", kernel_regularizer=l2(l2=0.01))(out)
        out = Dropout(0.5)(out)
        out = Dense(32, activation="relu", kernel_regularizer=l2(l2=0.01))(out)
        out = Dropout(0.2)(out)

        outputs = keras.layers.Dense(units=1, activation='sigmoid')(out)

        model = keras.Model(self.model.input, outputs, name='T2')
        # mm.summary()

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['acc', tf.keras.metrics.AUC(name='auc')])
        history = model.fit_generator(train_data,
                                      steps_per_epoch=train_num / self.batch_size,
                                      validation_data=valid_data,
                                      validation_steps=valid_num / self.batch_size,
                                      epochs=500,
                                      workers=2,
                                      class_weight={0: 0.2, 1: 0.8},
                                      callbacks=[self.lr_decay, self.save_model, self.early_stop])

        return model


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

    train_T2 = TrainDCNN()
    train_T2.training(train_set, y_train, val_set, y_val)
