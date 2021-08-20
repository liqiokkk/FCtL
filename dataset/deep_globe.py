import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "tif"])

def classToRGB(dataset, label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)
    if dataset == 1:
        pass
    else:
        indices = np.where(label == 1)
        colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
        indices = np.where(label == 0)
        colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    transform = ToTensor();
    #     plt.imshow(colmap)
    #     plt.show()
    return transform(colmap)


def class_to_target(inputs, numClass):
    batchSize, l, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, l, w, numClass), dtype=np.float32)
    for index in range(numClass):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=numClass, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2)


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs


class DeepGlobe(data.Dataset):
    """input and label image dataset"""

    def __init__(self, dataset, root, ids, label=False, transform=False):
        super(DeepGlobe, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.dataset = dataset
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)

    def __getitem__(self, index):
        sample = {}
        sample['id'] = self.ids[index][:-8]
        image = Image.open(os.path.join(self.root, "Sat/" + self.ids[index])) # w, h
        sample['image'] = image
        if self.label:
            if self.dataset == 1 :
                pass
            else:
                label = Image.open(os.path.join(self.root, 'Label/' + self.ids[index].replace('_sat.tif', '_mask.png')))
            sample['label'] = label
        if self.transform and self.label:
            image, label = self._transform(image, label)
            sample['image'] = image
            sample['label'] = label
        return sample

    def _transform(self, image, label):
        
        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        if np.random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, degree)
            label = transforms.functional.rotate(label, degree)

        
        return image, label


    def __len__(self):
        return len(self.ids)
