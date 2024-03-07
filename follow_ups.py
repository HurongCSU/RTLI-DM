import os
import random
from glob import glob
import tensorflow as tf

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.regularizers import l2

from data_organize.utils import get_split, get_images, get_performance, data_loader_tln

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7
num_fea = 64
if __name__ == '__main__':

    model = load_model('./model/T2_20231212_tln_roi_eff.h5')
    last_layer_outputs = tf.keras.Model(model.input, model.layers[-5].output)

    follow_up = glob('/home/hurong/3Unet_data/follow_ups/n4' + '/*_n4.nii')
    path = '/home/hurong/3Unet_data/follow_ups/20231213/TLN_slices'

    features_train = []
    train_y = []
    for name in follow_up:
        print(os.path.basename(name).replace('.nii', ''))
        tmp = glob(os.path.join(path, os.path.basename(name).replace('.nii', '') + '*.png'))
        imgs, _ = data_loader_tln(tmp, [0] * len(tmp))
        features_ = np.zeros((30 * num_fea))
        if len(imgs) == 0:
            features_train.append(features_)
            continue
        features = last_layer_outputs.predict(imgs)
        features = features.flatten()
        features_[0:len(features)] = features
        features_train.append(features_)
        print(features.shape)
        print(len(features_train))

    features_train = np.array(features_train)
    fin_model = load_model('model/mlpb4_t2_5.h5')

    pred_train = fin_model.predict(features_train)

    for i, j in zip(follow_up, pred_train):
        print(i, j)
