# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image

class NeighborExtractor5(nn.Module):
    def __init__(self, input_channel):
        super(NeighborExtractor5, self).__init__()
        same_class_neighbor = np.array([[1., 1., 1., 1., 1.],
                                        [1., 1., 1., 1., 1.],
                                        [1., 1., 0., 1., 1.],
                                        [1., 1., 1., 1., 1.],
                                        [1., 1., 1., 1., 1.], ])
        same_class_neighbor = same_class_neighbor.reshape((1, 1, 5, 5))
        same_class_neighbor = np.repeat(same_class_neighbor, input_channel, axis=0)
        self.same_class_extractor = nn.Conv2d(input_channel, input_channel, kernel_size=5, padding=2, bias=False,
                                              groups=input_channel)
        self.same_class_extractor.weight.data = torch.from_numpy(same_class_neighbor)

    def forward(self, feat):
        output = self.same_class_extractor(feat)
        return output

class CBL(nn.Module):
    def __init__(self,num_classes = 2,weights = [2.0,0.1,0.5]):
        super(CBL,self).__init__()

        # 这里需要注意的是，conv_seg是最后一层网络
        self.num_classes = num_classes
        self.weights = weights
        base_weight = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1], ])
        base_weight = base_weight.reshape((1, 1, 5, 5))
        self.same_class_extractor_weight = np.repeat(base_weight, 256, axis=0)
        self.same_class_extractor_weight = torch.FloatTensor(self.same_class_extractor_weight)
        # self.same_class_extractor_weight.requires_grad(False)
        self.same_class_number_extractor_weight = base_weight
        self.same_class_number_extractor_weight = torch.FloatTensor(self.same_class_number_extractor_weight)

        '''
        本质就是计算论文中的公式（14）
        
       er_input: 从输入的形式来看，out['mask_features'],貌似是一个latent features,是的，它就是一个高维的features
       seg_label：应该就是正常的ground truth
       gt_boundary_seg：这里应该是通过某种方式计算出来的boundary的ground truth
       kernel_size=5：计算邻近像素的矩阵大小，根据论文原论述可知，这里应该是采用一个中空的固定卷积。
       '''
    def context_loss(self, er_input, seg_label, gt_boundary_seg, kernel_size=5):
        # 将seg_label的形状大小通过插值方式调整来与er_input相同
        seg_label = F.interpolate(seg_label.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        # 同理，也需要对boundary gt的形状进行调整
        gt_boundary_seg = F.interpolate(gt_boundary_seg.unsqueeze(1).float(), size=er_input.shape[2:],
                                        mode='nearest').long()
        context_loss_final = torch.tensor(0.0, device=er_input.device)
        context_loss = torch.tensor(0.0, device=er_input.device)
        # 获取参与运算的边缘像素mask gt_b (B,1,H,W)
        gt_b = gt_boundary_seg
        # 将ignore的像素置零，现在gt_b里面的0表示不参与loss计算
        gt_b[gt_b == 255] = 0
        seg_label_copy = seg_label.clone()
        seg_label_copy[seg_label_copy == 255] = 0
        gt_b = gt_b * seg_label_copy
        seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=256)[:, :, :, 0:self.num_classes].permute(0, 3,
                                                                                                                  1, 2)

        b, c, h, w = er_input.shape
        scale_num = b
        # 对batchsize中的每个对象进行操作
        for i in range(b):
            # 使用bool进行索引，方便后续进行像素position的选取
            cal_mask = (gt_b[i][0] > 0).bool()
            if cal_mask.sum() < 1:
                scale_num = scale_num - 1
                continue

            # 这里position是一个元组，分别代表非0元素索引的行列坐标，因此len(position[0])就代表非0元素的个数
            position = torch.where(gt_b[i][0])
            '''
            (kernel_size//2) <= position[0]：这个条件检查 x 坐标是否大于等于卷积核一半的尺寸。这是为了确保卷积核的中心不会超出图像的左边界。

            position[0] <= (er_input.shape[-2] - 1 - (kernel_size//2))：这个条件检查 x 坐标是否小于等于图像行数减去卷积核一半的尺寸。这是为了确保卷积核的中心不会超出图像的右边界。

            (kernel_size//2) <= position[1]：这个条件检查 y 坐标是否大于等于卷积核一半的尺寸。这是为了确保卷积核的中心不会超出图像的上边界。

            position[1] <= (er_input.shape[-1] - 1 - (kernel_size//2))：这个条件检查 y 坐标是否小于等于图像列数减去卷积核一半的尺寸。这是为了确保卷积核的中心不会超出图像的下边界。
            '''
            position_mask = ((kernel_size // 2) <= position[0]) * (
                    position[0] <= (er_input.shape[-2] - 1 - (kernel_size // 2))) * (
                                    (kernel_size // 2) <= position[1]) * (
                                    position[1] <= (er_input.shape[-1] - 1 - (kernel_size // 2)))
            position_selected = (position[0][position_mask], position[1][position_mask])
            position_shift_list = []  # 5*5-1 = 24
            # position_shift_list中每一个元素又是一个行列坐标tuple
            # 解释这里为什么能够在全图上起作用：这里的kernel_size = 5，因此有5*5-1 = 24个邻居，所以针对整个tensor而言
            # 就有24个视图，其中很关键的一步：position_mask的操作又恰好留出了空间，十分巧妙
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    if ki == kj == (kernel_size // 2):
                        continue
                    position_shift_list.append((position_selected[0] + ki - (kernel_size // 2),
                                                position_selected[1] + kj - (kernel_size // 2)))
            # context_loss_batchi = torch.zeros_like(er_input[i].permute(1,2,0)[position_selected][0])
            context_loss_pi = torch.tensor(0.0, device=er_input.device)
            for pi in range(len(position_shift_list)):
                boudary_simi = F.cosine_similarity(er_input[i].permute(1, 2, 0)[position_selected],
                                                   er_input[i].permute(1, 2, 0)[position_shift_list[pi]], dim=1)
                boudary_simi_label = torch.sum(
                    seg_label_one_hot[i].permute(1, 2, 0)[position_selected] * seg_label_one_hot[i].permute(1, 2, 0)[
                        position_shift_list[pi]], dim=-1)
                context_loss_pi = context_loss_pi + F.mse_loss(boudary_simi, boudary_simi_label.float())
            context_loss += (context_loss_pi / len(position_shift_list))
        context_loss = context_loss / scale_num
        if torch.isnan(context_loss):
            return context_loss_final
        else:
            return context_loss

    def er_loss4Semantic(self, er_input, seg_label, seg_logit, gt_boundary_seg,conv_seg_weight,conv_seg_bias):
        shown_class = list(seg_label.unique())
        pred_label = seg_logit.max(dim=1)[1]
        pred_label_one_hot = F.one_hot(pred_label, num_classes=self.num_classes).permute(0, 3, 1, 2)
        seg_label = F.interpolate(seg_label.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        gt_boundary_seg = F.interpolate(gt_boundary_seg.unsqueeze(1).float(), size=er_input.shape[2:],
                                        mode='nearest').long()
        # 获取参与运算的边缘像素mask gt_b (B,1,H,W)
        gt_b = gt_boundary_seg
        # 将ignore的像素置零，现在gt_b里面的0表示不参与loss计算
        gt_b[gt_b == 255] = 0
        edge_mask = gt_b.squeeze(1)
        # 下面按照每个出现的类计算每个类的er loss
        # 首先提取出每个类各自的boundary
        seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=256)[:, :, :, 0:self.num_classes].permute(0, 3,
                                                                                                                  1, 2)
        if self.same_class_extractor_weight.device != er_input.device:
            self.same_class_extractor_weight = self.same_class_extractor_weight.to(er_input.device)
            print("er move:", self.same_class_extractor_weight.device)
        if self.same_class_number_extractor_weight.device != er_input.device:
            self.same_class_number_extractor_weight = self.same_class_number_extractor_weight.to(er_input.device)
        # print(self.same_class_number_extractor_weight)
        same_class_extractor = NeighborExtractor5(256)
        # TODO
        same_class_extractor = same_class_extractor.to(er_input.device)
        same_class_extractor.same_class_extractor.weight.data = self.same_class_extractor_weight
        same_class_number_extractor = NeighborExtractor5(1)
        same_class_number_extractor = same_class_number_extractor.to(er_input.device)
        same_class_number_extractor.same_class_extractor.weight.data = self.same_class_number_extractor_weight

        try:
            shown_class.remove(torch.tensor(255))
        except:
            pass
        # er_input = er_input.permute(0,2,3,1)
        neigh_classfication_loss_total = torch.tensor(0.0, device=er_input.device)
        close2neigh_loss_total = torch.tensor(0.0, device=er_input.device)
        cal_class_num = len(shown_class)
        for i in range(len(shown_class)):
            # 获取当前类别的label掩膜
            # now_class_mask = seg_label_one_hot[:, shown_class[i], :, :]
            now_class_mask = seg_label_one_hot[:, shown_class[i].long(), :, :]
            # 获取当前类别预测的掩膜
            # now_pred_class_mask = pred_label_one_hot[:, shown_class[i], :, :]
            now_pred_class_mask = pred_label_one_hot[:, shown_class[i].long(), :, :]
            # er_input 乘当前类的mask，就把所有不是当前类的像素置为0了
            # 得到的now_neighbor_feat是只有当前类的特征
            now_neighbor_feat = same_class_extractor(er_input * now_class_mask.unsqueeze(1))
            # 获取当前邻居中为正样本的邻居的特征
            now_correct_neighbor_feat = same_class_extractor(
                er_input * (now_class_mask * now_pred_class_mask).unsqueeze(1))
            # 下面是获得当前类的每个像素邻居中同属当前点的像素个数
            now_class_num_in_neigh = same_class_number_extractor(now_class_mask.unsqueeze(1).float())
            # 获取邻居中为正样本的邻居个数
            now_correct_class_num_in_neigh = same_class_number_extractor(
                (now_class_mask * now_pred_class_mask).unsqueeze(1).float())
            # 需要得到 可以参与er loss 计算的像素
            # 一个像素若要参与er loss计算，需要具备：
            # 1.邻居中具有同属当前类的像素
            # 2.当前像素要在边界上
            # 如果没有满足这些条件的像素 则直接跳过这个类的计算
            pixel_cal_mask = (now_class_num_in_neigh.squeeze(1) >= 1) * (
                        edge_mask.bool() * now_class_mask.bool()).detach()
            pixel_mse_cal_mask = (now_correct_class_num_in_neigh.squeeze(1) >= 1) * (
                        edge_mask.bool() * now_class_mask.bool() * now_pred_class_mask.bool()).detach()
            if pixel_cal_mask.sum() < 1 or pixel_mse_cal_mask.sum() < 1:
                cal_class_num = cal_class_num - 1
                continue
            # 这里是把邻居特征做平均
            class_forward_feat = now_neighbor_feat / (now_class_num_in_neigh + 1e-5)
            # 把正确类别特征做平均
            class_correct_forward_feat = now_correct_neighbor_feat / (now_correct_class_num_in_neigh + 1e-5)

            # 选择出参与loss计算的像素的原始特征
            # origin_pixel_feat = er_input.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            origin_mse_pixel_feat = er_input.permute(0, 2, 3, 1)[pixel_mse_cal_mask].permute(1, 0).unsqueeze(
                0).unsqueeze(-1)
            # 选择出参与loss计算的像素的邻居平均特征
            neigh_pixel_feat = class_forward_feat.permute(0, 2, 3, 1)[pixel_cal_mask].permute(1, 0).unsqueeze(
                0).unsqueeze(-1)
            neigh_mse_pixel_feat = class_correct_forward_feat.permute(0, 2, 3, 1)[pixel_mse_cal_mask].permute(1,
                                                                                                              0).unsqueeze(
                0).unsqueeze(-1)
            # 邻居平均特征也要能够正确分类，且用同样的分类器才行
            # 这个地方可以试试不detach掉的
            neigh_pixel_feat_prediction = F.conv2d(neigh_pixel_feat,
                                                   weight=conv_seg_weight.to(neigh_pixel_feat.dtype).detach(),
                                                   bias=conv_seg_bias.to(neigh_pixel_feat.dtype).detach())
            # 为邻居平均特征的分类loss产生GT，即当前类中像素的邻居（因为都是同属当前类的邻居）的标签也是当前类
            # 因此乘shown_class[i]
            gt_for_neigh_output = shown_class[i] * torch.ones((1, neigh_pixel_feat_prediction.shape[2], 1)).to(
                er_input.device).long()
            # 对应原论文公式（11）
            # neigh_pixel_feat_prediction:float32   gt_for_neigh_output:float32

            neigh_classfication_loss = F.cross_entropy(neigh_pixel_feat_prediction, gt_for_neigh_output.long())
            # 当前点的像素 要向周围同类像素的平均特征靠近
            # close2neigh_loss = F.mse_loss(origin_pixel_feat, neigh_pixel_feat.detach())
            neigh_mse_pixel_feat_prediction = F.conv2d(neigh_mse_pixel_feat,
                                                       weight=conv_seg_weight.to(neigh_pixel_feat.dtype).detach(),
                                                       bias=conv_seg_bias.to(neigh_pixel_feat.dtype).detach())
            gt_for_neigh_mse_output = shown_class[i] * torch.ones((1, neigh_mse_pixel_feat_prediction.shape[2], 1)).to(
                er_input.device).long()
            neigh_classfication_loss = neigh_classfication_loss + F.cross_entropy(neigh_mse_pixel_feat_prediction,
                                                                                  gt_for_neigh_mse_output.long())

            # 对应原论文公式 （10）
            # 需要说明origin_mse_pixel_feat代表原始的features，而neigh_mse_pixel_feat是邻居的平均特征
            '''
            TODO 1、 origin_mse_pixel_feat和 neigh_mse_pixel_feat作为正样本对，执行对比学习,但还缺少负样本对
                -思路1：其他类别的邻居的类别特征分别都给提取出来，然后作为负样本对
            TODO 2、origin_mse_pixel_feat与邻居做对比学习，那么这里就有许多细节需要展开了
                - 正样本对构建问题：思路是与正样本邻居的特征做正样本对，有许多正样本的邻居，那该如何计算呢？
                    - 取所有的正样本邻居的特征均值做正样本
                    - 依次与每个邻居做正样本，然后取均值
                - 负样本对构建问题：思路是将邻居的其他类别的负样本都算作是负样本
                - 如果要考虑到假阴性样本，那该如何将他们纳入计算呢？
                
                这里有两种情况，一是邻居里面没有正样本，而是邻居里面没有负样本，如果出现这个情况，可以遵循上面提到的，跳过此次计算
            '''

            close2neigh_loss = F.mse_loss(origin_mse_pixel_feat, neigh_mse_pixel_feat.detach())
            neigh_classfication_loss_total = neigh_classfication_loss_total + neigh_classfication_loss
            close2neigh_loss_total = close2neigh_loss_total + close2neigh_loss
        if cal_class_num == 0:
            return neigh_classfication_loss_total, close2neigh_loss_total
        # 对应原论文公式（11）
        neigh_classfication_loss_total = neigh_classfication_loss_total / cal_class_num
        # 对应原论文公式 （10）
        close2neigh_loss_total = close2neigh_loss_total / cal_class_num
        return neigh_classfication_loss_total, close2neigh_loss_total

    def gt2boundary(self, gt, ignore_label=-1,boundary_width = 5):  # gt NHW
        gt_ud = gt[:, boundary_width:, :] - gt[:, :-boundary_width, :]  # NHW
        gt_lr = gt[:, :, boundary_width:] - gt[:, :, :-boundary_width]
        gt_ud = torch.nn.functional.pad(gt_ud, [0, 0, 0, boundary_width, 0, 0], mode='constant', value=0) != 0
        gt_lr = torch.nn.functional.pad(gt_lr, [0, boundary_width, 0, 0, 0, 0], mode='constant', value=0) != 0
        gt_combine = gt_lr + gt_ud
        del gt_lr
        del gt_ud

        # set 'ignore area' to all boundary
        gt_combine += (gt == ignore_label)

        return gt_combine > 0
    '''
     结合mmsegmentation框架代码学习，豁然开朗，得出如下结论：
     er_input：实际上代表了整个分割网络的“倒数第二层”，为什么说是倒数第二层？而不是backbone的features呢？
     如果是backbone的features，一开始我就觉得不对劲，因为这样去优化是很不合理的。
     后面通过阅读代码可知：整个分割网络可以分为两个部分
        - backbone：resnet50
        - model：DeepLabV3+，又可以将所有的model都看成两个部分
            - 最后一层：conv_seg，又称classifier，这一卷积层是用于最终的预测分割，因为它的输入通道数都还是很高维度的，比如256，输出维度就是类别数了
            - 除最后一层以外的部分
     seg_label：语义分割的标签
     gt_boundary_seg：语义分割的边界标签
    '''
    def forward(self, outputs,gt_sem = None,conv_seg_weight = None,conv_seg_bias = None):
        gt_sem_boundary = self.gt2boundary(gt_sem.squeeze())
        # gt_sem_boundary = gt_sem_boundary.unsqueeze(1)
        er_loss = self.er_loss4Semantic(
                   outputs['out_fuse'],
                   seg_label=gt_sem,
                   seg_logit=outputs['out_classifier'],
                   gt_boundary_seg=gt_sem_boundary,
                   conv_seg_weight = conv_seg_weight,
                   conv_seg_bias = conv_seg_bias
               )
        loss_A2C_SCE,loss_A2C_pair = er_loss[0],er_loss[1]
        if self.weights[2] == 0.0:
            return self.weights[0] * (self.weights[1] * loss_A2C_SCE + loss_A2C_pair)
        else:
            loss_A2PN = self.context_loss(
                        outputs['out_fuse'],
                        seg_label=gt_sem,
                        gt_boundary_seg=gt_sem_boundary)
            return self.weights[0] * (self.weights[2] * loss_A2PN + self.weights[1] * loss_A2C_SCE + loss_A2C_pair)




class CCBL(nn.Module):
    def __init__(self,num_classes = 2,weights = [2.0,0.1,0.5]):
        super(CBL,self).__init__()

        # 这里需要注意的是，conv_seg是最后一层网络
        self.num_classes = num_classes
        self.weights = weights
        base_weight = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1], ])
        base_weight = base_weight.reshape((1, 1, 5, 5))
        self.same_class_extractor_weight = np.repeat(base_weight, 256, axis=0)
        self.same_class_extractor_weight = torch.FloatTensor(self.same_class_extractor_weight)
        # self.same_class_extractor_weight.requires_grad(False)
        self.same_class_number_extractor_weight = base_weight
        self.same_class_number_extractor_weight = torch.FloatTensor(self.same_class_number_extractor_weight)

    def er_loss4Semantic(self, er_input, seg_label, seg_logit, gt_boundary_seg,conv_seg_weight,conv_seg_bias):
        shown_class = list(seg_label.unique())
        pred_label = seg_logit.max(dim=1)[1]
        pred_label_one_hot = F.one_hot(pred_label, num_classes=self.num_classes).permute(0, 3, 1, 2)
        seg_label = F.interpolate(seg_label.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        gt_boundary_seg = F.interpolate(gt_boundary_seg.unsqueeze(1).float(), size=er_input.shape[2:],
                                        mode='nearest').long()
        # 获取参与运算的边缘像素mask gt_b (B,1,H,W)
        gt_b = gt_boundary_seg
        # 将ignore的像素置零，现在gt_b里面的0表示不参与loss计算
        gt_b[gt_b == 255] = 0
        edge_mask = gt_b.squeeze(1)
        # 下面按照每个出现的类计算每个类的er loss
        # 首先提取出每个类各自的boundary
        seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=256)[:, :, :, 0:self.num_classes].permute(0, 3,
                                                                                                                  1, 2)
        if self.same_class_extractor_weight.device != er_input.device:
            self.same_class_extractor_weight = self.same_class_extractor_weight.to(er_input.device)
            print("er move:", self.same_class_extractor_weight.device)
        if self.same_class_number_extractor_weight.device != er_input.device:
            self.same_class_number_extractor_weight = self.same_class_number_extractor_weight.to(er_input.device)
        # print(self.same_class_number_extractor_weight)
        same_class_extractor = NeighborExtractor5(256)
        # TODO
        same_class_extractor = same_class_extractor.to(er_input.device)
        same_class_extractor.same_class_extractor.weight.data = self.same_class_extractor_weight
        same_class_number_extractor = NeighborExtractor5(1)
        same_class_number_extractor = same_class_number_extractor.to(er_input.device)
        same_class_number_extractor.same_class_extractor.weight.data = self.same_class_number_extractor_weight

        try:
            shown_class.remove(torch.tensor(255))
        except:
            pass
        # er_input = er_input.permute(0,2,3,1)
        neigh_classfication_loss_total = torch.tensor(0.0, device=er_input.device)
        close2neigh_loss_total = torch.tensor(0.0, device=er_input.device)
        cal_class_num = len(shown_class)
        for i in range(len(shown_class)):
            # 获取当前类别的label掩膜，获取另外两个类别的label mask
            now_class_mask = seg_label_one_hot[:, shown_class[i].long(), :, :]
            pre_class_mask = seg_label_one_hot[:, shown_class[(i - 1) % len(shown_class)].long(), :, :]
            post_class_mask = seg_label_one_hot[:, shown_class[(i + 1) % len(shown_class)].long(), :, :]
            # 获取当前类别预测的掩膜以及其他类别的掩膜
            now_pred_class_mask = pred_label_one_hot[:, shown_class[i].long(), :, :]
            pre_pred_class_mask = pred_label_one_hot[:, shown_class[(i - 1) % len(shown_class)].long(), :, :]
            post_pred_class_mask = pred_label_one_hot[:, shown_class[(i + 1) % len(shown_class)].long(), :, :]

            # er_input 乘当前类的mask，就把所有不是当前类的像素置为0了
            # 得到的now_neighbor_feat是只有当前类的特征
            now_neighbor_feat = same_class_extractor(er_input * now_class_mask.unsqueeze(1))
            pre_neighbor_feat = same_class_extractor(er_input * pre_class_mask.unsqueeze(1))
            post_neighbor_feat = same_class_extractor(er_input * post_class_mask.unsqueeze(1))
            # 获取当前邻居中为正样本的邻居的特征
            now_correct_neighbor_feat = same_class_extractor(
                er_input * (now_class_mask * now_pred_class_mask).unsqueeze(1))
            # 获取当前邻居中为负样本的其他类别的特征
            pre_correct_neighbor_feat = same_class_extractor(
                er_input * (pre_class_mask * pre_pred_class_mask).unsqueeze(1))
            post_correct_neighbor_feat = same_class_extractor(
                er_input * (post_class_mask * post_pred_class_mask).unsqueeze(1))
            # 下面是获得当前类的每个像素邻居中同属当前点的像素个数,以及邻居中其他类别的像素个数
            now_class_num_in_neigh = same_class_number_extractor(now_class_mask.unsqueeze(1).float())
            pre_class_num_in_neigh = same_class_number_extractor(pre_class_mask.unsqueeze(1).float())
            post_class_num_in_neigh = same_class_number_extractor(post_class_mask.unsqueeze(1).float())
            # 获取邻居中为正样本的邻居个数，以及其他类别预测对的邻居个数
            now_correct_class_num_in_neigh = same_class_number_extractor(
                (now_class_mask * now_pred_class_mask).unsqueeze(1).float())
            pre_correct_class_num_in_neigh = same_class_number_extractor(
                (pre_class_mask * pre_pred_class_mask).unsqueeze(1).float())
            post_correct_class_num_in_neigh = same_class_number_extractor(
                (post_class_mask * post_pred_class_mask).unsqueeze(1).float())
            ################### er loss的计算过程 ###############################
            # 需要得到 可以参与er loss 计算的像素
            # 一个像素若要参与er loss计算，需要具备：
            # 1.邻居中具有同属当前类的像素
            # 2.当前像素要在边界上
            # 3.邻居中具有不同属当前类的像素
            # 如果没有满足这些条件的像素 则直接跳过这个类的计算
            pixel_cal_mask = (now_class_num_in_neigh.squeeze(1) >= 1) * (
                    edge_mask.bool() * now_class_mask.bool()).detach()
            pixel_mse_cal_mask = (now_correct_class_num_in_neigh.squeeze(1) >= 1) * (
                    edge_mask.bool() * now_class_mask.bool() * now_pred_class_mask.bool()).detach()
            pre_pixel_cal_mask = (pre_correct_class_num_in_neigh.squeeze(1) >= 1) * (
                    edge_mask.bool() * pre_class_mask.bool() * pre_pred_class_mask.bool()).detach()
            post_pixel_cal_mask = (post_correct_class_num_in_neigh.squeeze(1) >= 1) * (
                    edge_mask.bool() * post_class_mask.bool() * post_pred_class_mask.bool()).detach()
            if pixel_cal_mask.sum() < 1 or pixel_mse_cal_mask.sum() < 1 or pre_pixel_cal_mask.sum() < 1 or post_pixel_cal_mask.sum() < 1:
                cal_class_num = cal_class_num - 1
                continue
            # 这里是把同类别的邻居特征做平均
            class_forward_feat = now_neighbor_feat / (now_class_num_in_neigh + 1e-5)
            # 把正确同类别邻居特征做平均，
            class_correct_forward_feat = now_correct_neighbor_feat / (now_correct_class_num_in_neigh + 1e-5)

            #############################这里是我额外加的，为了计算对比学习
            pre_class_forward_feat = pre_neighbor_feat / (pre_class_num_in_neigh + 1e-5)
            pre_class_correct_forward_feat = pre_correct_neighbor_feat / (pre_correct_class_num_in_neigh + 1e-5)
            post_class_forward_feat = post_neighbor_feat / (post_class_num_in_neigh + 1e-5)
            post_class_correct_forward_feat = post_correct_neighbor_feat / (post_correct_class_num_in_neigh + 1e-5)
            ####################################

            # 选择出参与erloss计算的像素的原始特征
            # origin_pixel_feat = er_input.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            origin_mse_pixel_feat = er_input.permute(0, 2, 3, 1)[pixel_mse_cal_mask].permute(1, 0).unsqueeze(
                0).unsqueeze(-1)
            # 选择出参与loss计算的像素的邻居平均特征
            neigh_pixel_feat = class_forward_feat.permute(0, 2, 3, 1)[pixel_cal_mask].permute(1, 0).unsqueeze(
                0).unsqueeze(-1)
            # 当前类别预测正确的邻居的特征
            neigh_mse_pixel_feat = class_correct_forward_feat.permute(0, 2, 3, 1)[pixel_mse_cal_mask].permute(1,
                                                                                                              0).unsqueeze(
                0).unsqueeze(-1)
            # 邻居平均特征也要能够正确分类，且用同样的分类器才行
            # 这个地方可以试试不detach掉的
            # 为什么要传入卷积呢，因为传入进去的特征是邻居里当前类别的所有特征，因此就有正确的和假阳性，所以需要利用分类头来分类一下
            neigh_pixel_feat_prediction = F.conv2d(neigh_pixel_feat,
                                                   weight=conv_seg_weight.to(neigh_pixel_feat.dtype).detach(),
                                                   bias=conv_seg_bias.to(neigh_pixel_feat.dtype).detach())
            # 为邻居平均特征的分类loss产生GT，即当前类中像素的邻居（因为都是同属当前类的邻居）的标签也是当前类
            # 因此乘shown_class[i]
            gt_for_neigh_output = shown_class[i] * torch.ones((1, neigh_pixel_feat_prediction.shape[2], 1)).to(
                er_input.device).long()
            # 对应原论文公式（11）
            # neigh_pixel_feat_prediction:float32   gt_for_neigh_output:float32

            neigh_classfication_loss = F.cross_entropy(neigh_pixel_feat_prediction, gt_for_neigh_output.long())
            # 当前点的像素 要向周围同类像素的平均特征靠近
            # close2neigh_loss = F.mse_loss(origin_pixel_feat, neigh_pixel_feat.detach())
            neigh_mse_pixel_feat_prediction = F.conv2d(neigh_mse_pixel_feat,
                                                       weight=conv_seg_weight.to(neigh_pixel_feat.dtype).detach(),
                                                       bias=conv_seg_bias.to(neigh_pixel_feat.dtype).detach())
            gt_for_neigh_mse_output = shown_class[i] * torch.ones((1, neigh_mse_pixel_feat_prediction.shape[2], 1)).to(
                er_input.device).long()
            neigh_classfication_loss = neigh_classfication_loss + F.cross_entropy(neigh_mse_pixel_feat_prediction,
                                                                                  gt_for_neigh_mse_output.long())

            # 对应原论文公式 （10）
            # 需要说明origin_mse_pixel_feat代表原始的features，而neigh_mse_pixel_feat是邻居的平均特征
            '''
            TODO 1、 origin_mse_pixel_feat和 neigh_mse_pixel_feat作为正样本对，执行对比学习,但还缺少负样本对
                -思路1：其他类别的邻居的类别特征分别都给提取出来，然后作为负样本对
            TODO 2、origin_mse_pixel_feat与邻居做对比学习，那么这里就有许多细节需要展开了
                - 正样本对构建问题：思路是与正样本邻居的特征做正样本对，有许多正样本的邻居，那该如何计算呢？
                    - 取所有的正样本邻居的特征均值做正样本
                    - 依次与每个邻居做正样本，然后取均值
                - 负样本对构建问题：思路是将邻居的其他类别的负样本都算作是负样本
                - 如果要考虑到假阴性样本，那该如何将他们纳入计算呢？

                这里有两种情况，一是邻居里面没有正样本，而是邻居里面没有负样本，如果出现这个情况，可以遵循上面提到的，跳过此次计算
            '''

            close2neigh_loss = F.mse_loss(origin_mse_pixel_feat, neigh_mse_pixel_feat.detach())
            neigh_classfication_loss_total = neigh_classfication_loss_total + neigh_classfication_loss
            close2neigh_loss_total = close2neigh_loss_total + close2neigh_loss
        if cal_class_num == 0:
            return neigh_classfication_loss_total, close2neigh_loss_total
        # 对应原论文公式（11）
        neigh_classfication_loss_total = neigh_classfication_loss_total / cal_class_num
        # 对应原论文公式 （10）
        close2neigh_loss_total = close2neigh_loss_total / cal_class_num
        return neigh_classfication_loss_total, close2neigh_loss_total

    def gt2boundary(self, gt, ignore_label=-1,boundary_width = 5):  # gt NHW
        gt_ud = gt[:, boundary_width:, :] - gt[:, :-boundary_width, :]  # NHW
        gt_lr = gt[:, :, boundary_width:] - gt[:, :, :-boundary_width]
        gt_ud = torch.nn.functional.pad(gt_ud, [0, 0, 0, boundary_width, 0, 0], mode='constant', value=0) != 0
        gt_lr = torch.nn.functional.pad(gt_lr, [0, boundary_width, 0, 0, 0, 0], mode='constant', value=0) != 0
        gt_combine = gt_lr + gt_ud
        del gt_lr
        del gt_ud

        # set 'ignore area' to all boundary
        gt_combine += (gt == ignore_label)

        return gt_combine > 0
    '''
     结合mmsegmentation框架代码学习，豁然开朗，得出如下结论：
     er_input：实际上代表了整个分割网络的“倒数第二层”，为什么说是倒数第二层？而不是backbone的features呢？
     如果是backbone的features，一开始我就觉得不对劲，因为这样去优化是很不合理的。
     后面通过阅读代码可知：整个分割网络可以分为两个部分
        - backbone：resnet50
        - model：DeepLabV3+，又可以将所有的model都看成两个部分
            - 最后一层：conv_seg，又称classifier，这一卷积层是用于最终的预测分割，因为它的输入通道数都还是很高维度的，比如256，输出维度就是类别数了
            - 除最后一层以外的部分
     seg_label：语义分割的标签
     gt_boundary_seg：语义分割的边界标签
    '''
    def forward(self, outputs,gt_sem = None,conv_seg_weight = None,conv_seg_bias = None):
        gt_sem_boundary = self.gt2boundary(gt_sem.squeeze())
        # gt_sem_boundary = gt_sem_boundary.unsqueeze(1)
        er_loss = self.er_loss4Semantic(
            outputs['out_fuse'],
            seg_label=gt_sem,
            seg_logit=outputs['out_classifier'],
            gt_boundary_seg=gt_sem_boundary,
            conv_seg_weight = conv_seg_weight,
            conv_seg_bias = conv_seg_bias
        )
        loss_A2C_SCE,loss_A2C_pair = er_loss[0],er_loss[1]
        if self.weights[2] == 0.0:
            return self.weights[0] * (self.weights[1] * loss_A2C_SCE + loss_A2C_pair)
        else:
            loss_A2PN = self.context_loss(
                outputs['out_fuse'],
                seg_label=gt_sem,
                gt_boundary_seg=gt_sem_boundary)
            return self.weights[0] * (self.weights[2] * loss_A2PN + self.weights[1] * loss_A2C_SCE + loss_A2C_pair)



if __name__ == '__main__':
    from segment.losses.abl import ABL
    import time
    num_classes = 3
    cbl = CBL(num_classes)
    outputs = {'out_fuse':torch.randn(4,256,64,64).to('cuda'),
               'out_classifier':torch.randn(4,3,64,64).to('cuda')
               }
    seg_label = torch.zeros(4,512,512).to('cuda')
    seg_label[0,5:100] = 1.0
    seg_label[0,100:200] = 1.0
    seg_label[0,205:300] = 2.0
    seg_label[0,300:400] = 2.0
    from segment.modules.semseg.deeplabv3plus import DeepLabV3Plus
    model = DeepLabV3Plus('resnet50', num_classes)
    model = model.to('cuda')
    start_time = time.time()
    loss = cbl(outputs = outputs,gt_sem = seg_label,conv_seg_weight = model.classifier.weight,conv_seg_bias = model.classifier.bias)
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time) # 2s




