from glob import glob
import albumentations

import numpy as np
import pandas as pd
import keras

import matplotlib.pyplot as plt
import tensorflow as tf

from keras.losses import binary_crossentropy
import keras.callbacks as callbacks

import glob
import os
import random
from PIL import Image

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train_mask = np.load('/home/hurong/PycharmProjects/keras-seg/code/train_you_mask.npy')
train_no_mask = np.load('/home/hurong/PycharmProjects/keras-seg/code/train_no_mask.npy')

train_mask = np.concatenate((train_mask, np.random.choice(train_no_mask, len(train_mask))))
val_mask = np.load('/home/hurong/PycharmProjects/keras-seg/code/val_you_mask.npy')
print(val_mask.shape)
# val_mask = np.concatenate((val_mask, np.load('/home/hurong/PycharmProjects/keras-seg/code/val_no_mask.npy')))

train_mask = [i.replace('..', '/home/hurong/PycharmProjects/keras-seg') for i in train_mask]
val_mask = [i.replace('..', '/home/hurong/PycharmProjects/keras-seg') for i in val_mask]

h, w, batch_size = 256, 256, 1


class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 mask_path, augmentations=None, batch_size=batch_size, img_size=256, n_channels=3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.train_im_paths = mask_path
        # self.train_mask_paths = train_mask

        self.img_size = img_size

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.train_im_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min((index + 1) * self.batch_size, len(self.train_im_paths))]

        # Find list of IDs
        list_IDs_im = [self.train_im_paths[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X, np.array(y) / 255
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im), np.array(mask) / 255

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_im_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im), self.img_size, self.img_size, self.n_channels))
        y = np.empty((len(list_IDs_im), self.img_size, self.img_size, 1))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):

            im = np.array(Image.open(im_path.replace('masks', 'images')))
            mask = np.array(Image.open(im_path))

            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)

            #             # Resize sample
            X[i,] = cv2.resize(im, (self.img_size, self.img_size))

            # Store class
            y[i,] = cv2.resize(mask, (self.img_size, self.img_size))[..., np.newaxis]
            y[y > 0] = 255

        return np.uint8(X), np.uint8(y)


img_size = 256
import cv2
import albumentations as albu

AUGMENTATIONS_TRAIN = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

    albu.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0),
    albu.RandomCrop(height=img_size, width=img_size, always_apply=True),

    # albu.GaussNoise(p=0.2),
    albu.Perspective(p=0.5),

    albu.OneOf(
        [
            albu.CLAHE(p=0.5),
        ],
        p=1,
    ),

    albu.OneOf(
        [
            albu.Sharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
            albu.RandomBrightnessContrast(p=1),
            albu.RandomGamma(p=1),
        ],
        p=0.9,
    ),

    albu.OneOf(
        [
            albu.RandomBrightnessContrast(p=1),
            albu.HueSaturationValue(p=1),
        ],
        p=0.9,
    ),
])

AUGMENTATIONS_TEST = albu.Compose([albu.PadIfNeeded(img_size, img_size)])

a = DataGenerator(val_mask,batch_size=64, shuffle=False)
images, masks = a.__getitem__(0)
max_images = 64
grid_width = 16
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))



# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow


def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union

        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.compat.v1.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

from keras import backend as K
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


from segmentation_models import Unet, Nestnet, Xnet

model = Nestnet(backbone_name='vgg16', encoder_weights='imagenet', decoder_block_type='transpose')
model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[my_iou_metric])
epochs = 70
batch_size = 16
# Generators
training_generator = DataGenerator(train_mask,augmentations=AUGMENTATIONS_TRAIN, img_size=img_size)
validation_generator = DataGenerator(val_mask,
    augmentations=AUGMENTATIONS_TEST,
    img_size=img_size)

ALPHA = 0.8
GAMMA = 2

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss


model = keras.models.load_model('./keras/model.h5',custom_objects={'FocalLoss':FocalLoss,'dice_coef':dice_coef})
for i in range((len(validation_generator))):
    img, gt_mask = validation_generator[i]
    tln_mask = model.predict(img)
    # pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    # print(pr_mask.shape)
    # pr_mask[pr_mask >= 0.5] = 1
    # pr_mask[pr_mask < 0.5] = 0

    tln_mask[tln_mask >= 0.5] = 1
    tln_mask[tln_mask < 0.5] = 0
    # threshold = 20
    # if np.max(pr_mask) == 0:
    #     continue
    #
    plt.subplot(1, 3, 1)
    brain = tln_mask[0]
    plt.imshow(brain, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(gt_mask))
    plt.subplot(1, 3, 3)
    plt.imshow(img[0])
    plt.show()
    # print(train_mask[i])


