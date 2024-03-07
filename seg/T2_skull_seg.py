import gc
import os
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import cv2
import albumentations as albu
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import torch
from PIL import Image
import segmentation_models_pytorch as smp

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30"

IMAGE_SIZE = 256


def normalize(image):
    max_val = np.max(image)
    min_val = np.min(image)
    image = (image - min_val) / (max_val - min_val) * 255
    image = np.asarray(image, dtype=np.uint8)
    return image


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):
    def __init__(
            self,
            img_dir,
            augmentation=None,
            preprocessing=None,
            train_flag=False
    ):
        self.masks_fps = [i.replace('images', 'skull_man') for i in img_dir]
        self.images_fps = img_dir
        self.class_values = [1]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.train_flag = train_flag

    def __getitem__(self, i):

        # read data
        image = Image.open(self.images_fps[i])
        image = np.squeeze(image)
        image = normalize(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_LINEAR))
        image = np.stack((image, image, image), axis=2)
        mask = Image.open(self.masks_fps[i])
        mask = np.squeeze(mask)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST)
        mask = mask[:, :, np.newaxis]

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)


SIZE = 256


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),

        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=SIZE, min_width=SIZE, always_apply=True, border_mode=0),
        albu.RandomCrop(height=SIZE, width=SIZE, always_apply=True),

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
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def to_tensor_mask(x, **kwargs):
    return x.transpose(2, 0, 1).astype('int32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor_mask)
    ]
    return albu.Compose(_transform)


def main():
    train_mask = np.load('../data_organize/T2_train_skull.npy')
    val_mask = np.load('../data_organize/T2_val_skull.npy')

    print(val_mask.shape)

    ENCODER = 'vgg11_bn'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None  # could be None for logits or 'softmax2d' for multiclass segmentation


    DEVICE = 'cuda'

    seed = 7

    augmented_dataset = Dataset(
        train_mask,
        augmentation=get_training_augmentation(),
        train_flag=True
    )

    for i in range(5):
        image, mask = augmented_dataset[i]
        visualize(
            image=image,
            cars_mask=mask
        )

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
    )
    model = torch.nn.DataParallel(model)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        train_mask,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        train_flag=True
    )

    valid_dataset = Dataset(
        val_mask,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        train_flag=False
    )

    test_dataset = Dataset(
        val_mask,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        train_flag=False
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    loss = smp.losses.FocalLoss(mode='binary', alpha=0.85, gamma=2)
    loss.__name__ = 'focal_loss'
    metrics = [
        smp.utils.metrics.Dice()
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.001),
    ])
    # RE_0919
    # model = torch.load('./T2_seg.pth')

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs

    max_score = 0

    for i in range(0, 200):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # test_logs = test_epoch.run(test_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['dice_score']:
            max_score = valid_logs['dice_score']
            torch.save(model, './model/T2_skull_seg.pth')
            print('Model saved!')

        # if i % 30 == 29:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
        #     print('Decrease decoder learning rate to ' + str(optimizer.param_groups[0]['lr'] * 0.5))

    # if torch.cuda.is_available():
    #     with torch.cuda.device('cuda:0'):
    #         torch.cuda.empty_cache()

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    model.eval()
    with torch.no_grad():
        logs = test_epoch.run(test_loader)


if __name__ == '__main__':
    main()
