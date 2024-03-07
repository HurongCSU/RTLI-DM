import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf
from keras.models import load_model

tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt


def get_heatmap(model, img_, last_conv_layer_name):
    pred = model.predict(img_)
    index = np.argmax(pred[0])
    print(index)
    output = model.output[:, index]
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_])
    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heat_map = np.mean(conv_layer_output_value, axis=-1)
    heat_map = np.maximum(heat_map, 0)
    heat_map /= np.max(heat_map)
    return heat_map, pred

NET_SIZE = 224


if __name__ == '__main__':
    K.set_learning_phase(False)
    model = load_model('./model/T2_20231212_tln_keep_dense.h5')
    # model.summary()
    path = '/home/hurong/PycharmProjects/RI2023/data/RI_slices/keep_dim'

    df = pd.read_excel('./fp_fns.xls', sheet_name='chenyi_fn')
    fp_list = df['ID'].values

    for i in fp_list:
        print(i)
        tmp = sorted(glob(os.path.join(path, 'chenyi_all', str(i) + '_t2*.png')))

        for k in tmp:
            im = cv2.imread(k)
            if np.max(im) == 0:
                continue
            im = cv2.resize(im, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
            im = im / np.max(im)
            image = im[np.newaxis, :, :, :]
            heatmap, prob = get_heatmap(model, image, 'conv5_block16_concat')
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[2]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            s_img = heatmap * 0.2 + im * 225

            cv2.imwrite('./heatmap/chenyi/fn/' + str(i) + '_' + str(prob) + 'heatmap.png', heatmap * 0.2)
            cv2.imwrite('./heatmap/chenyi/fn/' + str(i) + '_' + str(prob) + 'heatmap1.png', heatmap)
            cv2.imwrite('./heatmap/chenyi/fn/' + str(i) + '_' + str(prob) + '.png', s_img)