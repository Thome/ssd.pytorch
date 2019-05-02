from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
from .bp import BPDetection, BPAnnotationTransform, BP_CLASSES, BP_ROOT, imgToAnns
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    
    #sample[0]: tensor of size [3, 300, 300]
    #sample[1]: [annotation1, annotation2, ...]
    #annotation: [x1,y1,x2,y2,label]

    targets = []
    imgs = []
    for sample in batch:
        bbxs = []
        for anno in sample[1]:
            xyz = [0.0,0.0,0.0,0.0,0.0]
            xyz[0]= (float)(anno[0])
            xyz[1]= (float)(anno[1])
            xyz[2]= (float)(anno[2])
            xyz[3]= (float)(anno[3])
            xyz[4]= anno[4]
            bbxs.append(torch.FloatTensor(xyz))
        targets.append(torch.stack(bbxs, 0))
        imgs.append(sample[0])
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
