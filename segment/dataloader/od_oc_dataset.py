import torch

from segment.dataloader.transform import crop, hflip, normalize, resize, blur, cutout,dist_transform
from segment.dataloader.transform import random_rotate,random_translate,add_salt_pepper_noise,random_scale
import cv2
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import scipy.io

def get_labels(task,mask_path):
    '''
    :param task:
    :param org_mask: 原始的mask：0背景，1视盘，2视杯
    :return:
    '''
    if mask_path.endswith('mat'):
        org_mask = scipy.io.loadmat(mask_path)['maskFull']
        if task == 'od':
            org_mask[org_mask > 0] = 1
            return Image.fromarray(org_mask)
        elif task == 'oc':
            org_mask[org_mask == 1] = 0
            org_mask[org_mask == 2] = 1
            return Image.fromarray(org_mask)
        else:
            return Image.fromarray(org_mask)
    else:
        org_mask = Image.open(mask_path)
        org_mask = np.array(org_mask)

    mask = np.zeros_like(org_mask)
    if task == 'od':
        mask[org_mask > 0] = 1
        return Image.fromarray(mask)
    elif task == 'oc':
        mask[org_mask == 2] = 1
        return Image.fromarray(mask)
    else:
        return Image.fromarray(org_mask.astype(np.uint8))



class SemiDataset(Dataset):
    def __init__(self,task, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None,add_unlabeled_id_path=None, pseudo_mask_path=None,aug=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.aug = aug
        self.task = task
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.pseudo_mask_path = pseudo_mask_path

        self.add_unlabeled_ids = None

        '''
        细节，如果标记数据少于未标记数据，那么在此过程中，会自动复制样本，直到与未标记数量相当，因此100-700，会变成700-700
        '''
        if mode == 'semi_train':
            if labeled_id_path is not None:
                with open(labeled_id_path, 'r') as f:
                    self.labeled_ids = f.read().splitlines()
                self.ids = self.labeled_ids
            if unlabeled_id_path is not None:
                with open(unlabeled_id_path, 'r') as f:
                    self.unlabeled_ids = f.read().splitlines()
                self.ids = self.unlabeled_ids

            if add_unlabeled_id_path is not None:
                with open(add_unlabeled_id_path, 'r') as f:
                    self.add_unlabeled_ids = f.read().splitlines()
                self.unlabeled_ids.extend(self.add_unlabeled_ids)

            if labeled_id_path is not None and unlabeled_id_path is not None:
                    self.ids = \
                        self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        elif mode == 'src_tgt_train':
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids =  self.unlabeled_ids

        else:
            if mode == 'val':
                id_path = 'dataset/%s/val.txt' % name
            elif mode == 'test':
                id_path = 'dataset/%s/test.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path

            elif mode == 'train':
                id_path = labeled_id_path
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if add_unlabeled_id_path is not None and mode == 'label':
                with open(add_unlabeled_id_path, 'r') as f:
                    self.add_unlabeled_ids = f.read().splitlines()
                    self.ids.extend(self.add_unlabeled_ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img_path = os.path.join(self.root, id.split(' ')[0])
        # if img_path.endswith('.tif'):
        #     img = Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        # else:
        img = Image.open(img_path)
        mask_path = os.path.join(self.root, id.split(' ')[1])

        if self.mode == 'val' or self.mode == 'test' or self.mode == 'label':
            if self.add_unlabeled_ids is not None and id in self.add_unlabeled_ids:
                mask_arr = np.zeros(img.size)
                mask = Image.fromarray(mask_arr.astype(np.uint8))
            else:
                mask = get_labels(self.task,mask_path)
            img, mask = resize(img, mask, 512)
            img, mask = normalize(img, mask)
            boundary = dist_transform(mask)
            return {'img':img, 'mask':mask, 'id':id,'boundary':boundary}

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = get_labels(self.task, mask_path)

        else:
            fname = os.path.basename(id.split(' ')[1])
            mask = get_labels(self.task, os.path.join(self.pseudo_mask_path, fname))
        # basic augmentation on all training images
        # img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)
        img, mask = random_rotate(img = img,mask = mask,max_rotation_angle=90, p=0.5)
        img, mask = random_translate(img = img,mask = mask,max_translate_percent=0.15,  p=0.5)
        img, mask = add_salt_pepper_noise(img = img,mask = mask,noise_level=0.02, p=0.5)
        img, mask = random_scale(img = img,mask = mask,max_scale=1.2, p=0.5)


        img, mask = resize(img, mask, self.size)

        # strong augmentation on unlabeled images
        if (self.mode == 'semi_train' or self.mode == 'src_tgt_train' and id in self.unlabeled_ids) and self.aug :
            if self.aug.strong.Not:
                img, mask = normalize(img, mask)
                boundary = dist_transform(mask)
                return {'img':img, 'mask':mask,'boundary':boundary}

            if self.aug == None or self.aug.strong.default:
                if random.random() < 0.8:
                    img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
                img = transforms.RandomGrayscale(p=0.2)(img)
                img = blur(img, p=0.5)
                # img, mask = cutout(img, mask, p=0.5)
            if self.aug.strong.ColorJitter:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            if self.aug.strong.RandomGrayscale:
                img = transforms.RandomGrayscale(p=1.0)(img)
            if self.aug.strong.blur:
                img = blur(img, p=1.0)
            if self.aug.strong.cutout:
                img, mask = cutout(img, mask, p=1.0)

            img, mask = normalize(img, mask)
            boundary = dist_transform(mask)
            return {'img':img, 'mask':mask,'boundary':boundary}
        img, mask = normalize(img, mask)
        boundary = dist_transform(mask)
        return {'img':img, 'mask':mask,'boundary':boundary}

    def __len__(self):
        return len(self.ids)

class Validation(SemiDataset):
    def __init__(self,task, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None,add_unlabeled_id_path=None, pseudo_mask_path=None,aug=None):
        super().__init__(task=task,
                         name=name,
                         root=root,
                         mode=mode,
                         size=size,
                         labeled_id_path=labeled_id_path,
                         unlabeled_id_path=unlabeled_id_path,
                         add_unlabeled_id_path=add_unlabeled_id_path,
                         pseudo_mask_path=pseudo_mask_path,
                         aug=aug)

class SupTrain(SemiDataset):
    def __init__(self,task, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None,add_unlabeled_id_path=None, pseudo_mask_path=None,aug=None):
        super().__init__(task=task,
                         name=name,
                         root=root,
                         mode=mode,
                         size=size,
                         labeled_id_path=labeled_id_path,
                         unlabeled_id_path=unlabeled_id_path,
                         add_unlabeled_id_path=add_unlabeled_id_path,
                         pseudo_mask_path=pseudo_mask_path,
                         aug=aug)

class SemiTrain(SemiDataset):
    def __init__(self,task, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None,add_unlabeled_id_path=None, pseudo_mask_path=None,aug=None):
        super().__init__(task=task,
                         name=name,
                         root=root,
                         mode=mode,
                         size=size,
                         labeled_id_path=labeled_id_path,
                         unlabeled_id_path=unlabeled_id_path,
                         add_unlabeled_id_path=add_unlabeled_id_path,
                         pseudo_mask_path=pseudo_mask_path,
                         aug=aug)

class SemiUabledTrain(SemiDataset):
    def __init__(self,task, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None,add_unlabeled_id_path=None, pseudo_mask_path=None,aug=None):
        super().__init__(task=task,
                         name=name,
                         root=root,
                         mode=mode,
                         size=size,
                         labeled_id_path=labeled_id_path,
                         unlabeled_id_path=unlabeled_id_path,
                         add_unlabeled_id_path=add_unlabeled_id_path,
                         pseudo_mask_path=pseudo_mask_path,
                         aug=aug)