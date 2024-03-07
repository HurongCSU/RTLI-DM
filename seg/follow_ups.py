import gc
import os
from glob import glob

from skimage import morphology
from torch.nn import Sigmoid
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30"

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
        self.images_fps = [i.replace('masks', 'images') for i in img_dir]
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

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        return len(self.images_fps)


SIZE = 256


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


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
        albu.Lambda(image=to_tensor)
    ]
    return albu.Compose(_transform)


def main():
    train_mask = np.load('/home/hurong/3Unet_data/follow_ups/Slices/follow_ups.npy')
    val_mask = np.load('../data_organize/T2_val_skull.npy')

    ENCODER = 'vgg11_bn'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'

    seed = 7
    # re09191
    # model = torch.load('./model/T2_skull_seg.pth')
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module
    model_tln = torch.load('/home/hurong/PycharmProjects/torch_seg/re_09191.pth')

    test_dataset_vis = Dataset(
        train_mask,
        train_flag=False
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    test_dataset = Dataset(
        train_mask,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        train_flag=False
    )
    correct = 0
    total = 0
    sen = 0
    sen_total = 0
    import time
    start = time.time()
    for i in range(len(test_dataset_vis)):
        # n = np.random.choice(len(test_dataset))
        image_vis = test_dataset_vis[i]
        image = test_dataset[i]

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # pr_mask = model.predict(x_tensor)
        # pr_mask = Sigmoid()(pr_mask)
        tln_mask = model_tln.predict(x_tensor)
        # tln_mask = Sigmoid()(tln_mask)
        # pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        tln_mask = (tln_mask.squeeze().cpu().numpy())

        # print(pr_mask.shape)
        # pr_mask[pr_mask >= 0.5] = 1
        # pr_mask[pr_mask < 0.5] = 0

        tln_mask[tln_mask >= 0.5] = 1
        tln_mask[tln_mask < 0.5] = 0
        # threshold = 20
        # if np.max(pr_mask) == 0:
        #     continue
        #
        # plt.subplot(1, 3, 1)
        # brain = pr_mask * image_vis[:, :, 0]
        # plt.imshow(brain, cmap='gray')
        # plt.subplot(1, 3, 2)
        # plt.imshow(gt_mask)
        # plt.subplot(1, 3, 3)
        # plt.imshow(gt_mask * brain)
        # plt.show()
        # print(train_mask[i])
        # cv2.imwrite(train_mask[i].replace('images', 'skull_pred'), pr_mask,
        #             [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(train_mask[i].replace('images', 'masks'), tln_mask,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])

        """
        # extract boundary
        gray_image = cv2.cvtColor(pr_mask * 255, cv2.COLOR_GRAY2BGR)
        depth = gray_image.dtype
        if depth != np.uint8:
            print("Warning: Image depth is not CV_8U. Converting image to CV_8U.")
            gray_image = gray_image.astype(np.uint8)

        # 使用Canny边缘检测算法找到边界
        edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
        pr_mask = edges

        gray_image = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
        depth = gray_image.dtype
        if depth != np.uint8:
            print("Warning: Image depth is not CV_8U. Converting image to CV_8U.")
            gray_image = gray_image.astype(np.uint8)
        # 使用Canny边缘检测算法找到边界
        edges1 = cv2.Canny(gray_image, threshold1=30, threshold2=100)
        gt_mask = edges1

        contours_pr, _ = cv2.findContours(pr_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_vis, contours_pr, -1, (0, 255, 0), 1)  # 在感兴趣区域上绘制轮廓边界
        contours_gt, _ = cv2.findContours(gt_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_vis, contours_gt, -1, (0, 0, 255), 1)  # 在感兴趣区域上绘制轮廓边界

        # 显示结果图像
        plt.imshow(image_vis)
        plt.show()
        # visualize(
        #     image=image_vis,
        #     ground_truth_mask=mask,
        #     predicted_mask=pr_mask
        # )

        """
if __name__ == '__main__':
    main()
