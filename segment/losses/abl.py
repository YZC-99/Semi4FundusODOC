import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
# can find here: https://github.com/CoinCheung/pytorch-loss/blob/af876e43218694dc8599cc4711d9a5c5e043b1b2/label_smooth.py
from segment.losses.label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE
from segment.losses.korniadt import distance_transform
from torchvision import transforms
from functools import partial
from operator import itemgetter


# def color_map():
#     cmap = np.zeros((256, 3), dtype='uint8')
#     cmap[0] = np.array([0, 0, 0])
#     cmap[1] = np.array([255,0, 0])
#     cmap[2] = np.array([0, 255, 0])
#     cmap[3] = np.array([0, 0, 255])
#     return cmap
#
# cmap = color_map()
# Tools
def kl_div(a, b):  # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))


def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res


# Active Boundary Loss
class ABL(nn.Module):
    def __init__(self, isdetach=True, max_N_ratio=1 / 100, ignore_label=255, label_smoothing=0.0, weight=None,
                 max_clip_dist=20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach = isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LSSCE(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth=label_smoothing
            )

    def compute_dtm_gpu(self,img_gt, out_shape, kernel_size=5):
        """
        compute the distance transform map of foreground in binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the foreground Distance Map (SDM)
        dtm(x) = 0; x in segmentation boundary
                 inf|x-y|; x in segmentation
        """
        if len(out_shape) == 5:  # B,C,H,W,D
            fg_dtm = torch.cat([distance_transform(1 - img_gt[b].float(), kernel_size=kernel_size).unsqueeze(0) \
                                for b in range(out_shape[0])], axis=0)
        else:
            fg_dtm = distance_transform(1 - img_gt.float(), kernel_size=kernel_size)

        fg_dtm[~torch.isfinite(fg_dtm)] = kernel_size
        return fg_dtm

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h * w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr + kl_ud
        # 动态调整阈值
        while True:  # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *= 1.2
            else:
                break
        # dilate
        dilate_weight = torch.ones((1, 1, 3, 3)).cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_ud = gt[:, 1:, :] - gt[:, :-1, :]  # NHW
        gt_lr = gt[:, :, 1:] - gt[:, :, :-1]
        gt_ud = torch.nn.functional.pad(gt_ud, [0, 0, 0, 1, 0, 0], mode='constant', value=0) != 0
        gt_lr = torch.nn.functional.pad(gt_lr, [0, 1, 0, 0, 0, 0], mode='constant', value=0) != 0
        gt_combine = gt_lr + gt_ud
        del gt_lr
        del gt_ud

        # set 'ignore area' to all boundary
        gt_combine += (gt == ignore_label)

        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        # 找到非0的索引，返回的tenor很有意思：如(13958,3)，则代表有13958个非零元素，每个元素有三个维度的索引
        bound = torch.nonzero(pred_bound * 1)
        n, x, y = bound.T
        max_dis = 1e5

        logits = logits.permute(0, 2, 3, 1)  # NHWC

        # 给pred_dist_map周围填充max_dis
        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map, (1, 1, 1, 1, 0, 0), mode='constant',
                                                  value=max_dis)  # NH+2W+2

        logits_d = torch.nn.functional.pad(logits, (0, 0, 1, 1, 1, 1, 0, 0), mode='constant')  # N(H+2)(W+2)C
        logits_d[:, 0, :, :] = logits_d[:, 1, :, :]  # N(H+2)(W+2)C
        logits_d[:, -1, :, :] = logits_d[:, -2, :, :]  # N(H+2)(W+2)C
        logits_d[:, :, 0, :] = logits_d[:, :, 1, :]  # N(H+2)(W+2)C
        logits_d[:, :, -1, :] = logits_d[:, :, -2, :]  # N(H+2)(W+2)C

        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1, 0, 0, -1, 1, -1, 1, 0]
        y_range = [0, 0, -1, 1, 1, 1, -1, -1, 0]
        dist_maps = torch.zeros((0, len(x))).cuda()  # 8k
        kl_maps = torch.zeros((0, len(x))).cuda()  # 8k

        # 这里很有意思，上面使用bound.T获得的n,x,y，在这里居然可以进行像素索引的获取，这里就直接获取到了对应像素的logits，实在是妙
        kl_center = logits[(n, x, y)]  # KC

        for dx, dy in zip(x_range, y_range):
            '''
            这里也是使用之前获得的n，x，y进行元素索引，只不过加入了偏移量，这里很有意思,偏移量是以(0，0)作为起点，如下图，就是“4”为起点
                | 4| 0| 5|
                | 2| 8| 3|
                | 6| 1| 7|
                
            那么显然，8就是中心
        
            '''
            dist_now = pred_dist_map_d[(n, x + dx + 1, y + dy + 1)]
            dist_maps = torch.cat((dist_maps, dist_now.unsqueeze(0)), 0)

            '''
            当dx不等于0或者dy不等于0时计算kl_center与logits_now的kl散度在图中又是什么意思呢？
            具体表现为：不计算中心元素与center的kl散度，其实就是不计算自己与自己的kl散度，因为这样没有意义
            '''
            if dx != 0 or dy != 0:
                # 获取当前的logits_d
                logits_now = logits_d[(n, x + dx + 1, y + dy + 1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                # 这里的detach是一个很关键的操作
                # 如果进行detach，那么logits_now的梯度信息将不会保留，因此不会对模型的反向传播造成影响
                # 那么detach后，又对应了论文的那一个关键trick呢？
                # 通过阅读论文可知，此操作是为了解决梯度流的散度冲突，旨在阻断最小化梯度的反向传播
                if self.isdetach:
                    logits_now = logits_now.detach()
                # 计算kl_center和logits_now的kl散度
                kl_map_now = kl_div(kl_center, logits_now)

                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps, kl_map_now.unsqueeze(0)), 0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # 计算完成后的kl_maps应当有8个，因为是8个邻居
        # direction_gt shound be Nk  (8k->K)
        # 返回dist_maps距离最小的那一个的索引，这里很关键，返回的是距离最小的索引视图，也就是论文中所述的candidate pixel
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        # 根据坐标索引返回dist_map
        weight_ce = pred_dist_map[(n, x, y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        # 这里也很有意思，如果返回的索引是8，则代表是中间的那个元素，因此需要将他删除掉
        # direction_gt_idx 代表的是最小距离的像素索引
        direction_gt_idx = [direction_gt != 8]
        direction_gt = direction_gt[direction_gt_idx]

        kl_maps = torch.transpose(kl_maps, 0, 1)
        # direction_gt_idx 长度与kl_maps的一致
        # 注意，这里的direction_pred就是指与候选元素的kl散度
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out > 0, out, torch.zeros_like(out))

        return out

    def forward(self, logits, target):
        eps = 1e-10
        # 获得prediction的形状，logits：(b,c,h,w)
        ph, pw = logits.size(2), logits.size(3)
        # target:（b,h,w）
        h, w = target.size(1), target.size(2)

        # 如果logits和target的形状不一样，就调整形状大小
        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        # 获得gt_boundary
        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)
        # 获得dist_maps
        dist_maps = self.compute_dtm_gpu(gt_boundary.unsqueeze(0), logits.shape).cuda()
        # dist_maps = self.get_dist_maps(
        #     gt_boundary).cuda()  # <-- it will slow down the training, you can put it to dataloader.

        # 获得logtis的boundary
        pred_boundary = self.logits2boundary(logits)
        if pred_boundary.sum() < 1:  # avoid nan
            return None  # you should check in the outside. if None, skip this loss.

        # 获得gt和pred的direction
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary,
                                                                               logits)  # NHW,NHW,NCHW

        # direction_pred [K,8], direction_gt [K]
        # 一切都合理了

        '''
        个人理解
        论文公式 （4）：结合论文图2
        
        - 从图2可以看到，先是计算某一像素点的dist_map，然后选择当前像素邻居中距离最近的像素作为标签像素，
        论文认为距离最近的像素可以代表它的类别，对应论文中的direction_gt
        - 随后利用当前像素与它的邻居像素计算kl散度，kl散度计算出来再进行softmax就能得到一个由8个方向组成的概率分布
        - 然后根据这8个方向组成的概率分布与direction_gt计算交叉熵损失，这么做的目的是：将这个些像素统一推向direction_gt       
        '''
        loss = self.criterion(direction_pred, direction_gt)  # careful

        weight_ce = self.weight_func(weight_ce)
        loss = (loss * weight_ce).mean()  # add distance weight

        return loss


if __name__ == '__main__':
    from torch.backends import cudnn
    import os
    import random

    cudnn.benchmark = False
    cudnn.deterministic = True

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    n, c, h, w = 1, 2, 512, 512

    mask_path = './cropped_mask.png'
    gt_array = np.array(Image.open(mask_path))
    gt = torch.from_numpy(gt_array).long().unsqueeze(0).cuda()
    # gt[gt > 0] = 1
    # gt = torch.zeros((n, h, w)).cuda()
    # gt[0, 5] = 1
    # gt[0, 50] = 1
    logits = torch.randn((n, c, h, w)).cuda()

    abl = ABL()
    print(abl(logits, gt))