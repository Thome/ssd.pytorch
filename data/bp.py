from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

BP_CLASSES = (
    'arm','beard','ear','eye','face',
    'foot','hair','hand','head','leg',
    'mouth','nose','skull' )
#HOME= /home/thome
BP_ROOT = osp.join(HOME, "data/BP/")


def imgToAnns(annotations, img_id):
    found = 0
    anns = []
    for line in annotations:
        line = line.split(',')
        if (line[0] == img_id):
            found = 1
            anns.append(line)
        if (line[0] != img_id):
            if (not found):
                continue
            if (found):
                break
    return anns

class BPAnnotationTransform(object):

    def __call__(self, target, width, height):
        res = []
        for line in target:
            bndbox = []
            for i in [1,2,3,4]:
                point = int(line[i])
                # scale height or width
                point = point / width if i % 2 == 0 else point / height
                bndbox.append(point)
            bndbox.append(line[5]) #label
            res += [bndbox]
        return res

class BPDetection(data.Dataset):

    def __init__(self, root, transform=None, target_transform=BPAnnotationTransform(), dataset_name='BP'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self._imgpath = osp.join(root, 'images')
        self.ids = list()
        f = open(osp.join(BP_ROOT, "img_ids.txt"), 'r')
        for line in f:
            self.ids.append(line.rstrip('\n'))
        f.close()
        self.annotations = list()
        f = open(osp.join(BP_ROOT, "annotations.txt"), 'r')
        for line in f:
            self.annotations.append(line.rstrip('\n'))
        f.close()


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        path = osp.join(self._imgpath, img_id)
        img = cv2.imread(path)
        #print(img_id)
        #print(img.shape)
        height, width, channels = img.shape

        anno = imgToAnns(self.annotations, img_id)
        if self.target_transform is not None:
            anno = self.target_transform(anno, width, height)
            #print(anno)

        if self.transform is not None:
            anno = np.array(anno)
            boxes = anno[:, :4].astype(float)
            labels = np.array(anno[:, 4])
            #print(boxes)
            #print(labels)
            img, boxes, labels = self.transform(img, boxes, labels)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            anno = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), anno, height, width

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id)

    def pull_anno(self, index):
        img_id = self.ids[index]
        
        found = 0
        anno = []
        for line in self.annotations:
            line = line.split(',')
            if (line[0] == img_id):
                found = 1
                bbox = (line[5],(line[1],line[2],line[3],line[4])) #(class, (x1,y1,x2,y2))
                anno.append(bbox)
            if (line[0] != img_id):
                if (not found):
                    continue
                if (found):
                    break
        return img_id, anno

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)