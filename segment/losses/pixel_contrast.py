from segment.losses.info_nce import info_nce_loss,pixel_info_nce_loss
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
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


class ContrastCrossPixelCorrect(nn.Module):
    '''
    考虑到边界像素周围假阴样本过多的极端情况，现在针对local center做进一步的改进
    改进的目标是：希望周围的local center应该是分类正确的，而不应该带有分类失败的样本
    '''
    def __init__(self,num_classes = 2,extractor_channel = 256):
        super(ContrastCrossPixelCorrect,self).__init__()

        # 这里需要注意的是，conv_seg是最后一层网络
        self.num_classes = num_classes
        self.extractor_channel = extractor_channel

        base_weight = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1], ])
        base_weight = base_weight.reshape((1, 1, 5, 5))
        self.same_class_extractor_weight = np.repeat(base_weight, self.extractor_channel, axis=0)
        self.same_class_extractor_weight = torch.FloatTensor(self.same_class_extractor_weight)
        # self.same_class_extractor_weight.requires_grad(False)
        self.same_class_number_extractor_weight = base_weight
        self.same_class_number_extractor_weight = torch.FloatTensor(self.same_class_number_extractor_weight)

    def get_neigh(self,input,kernel_size = 3,pad = 1):
        b, c, h, w = input.size()
        input = input.reshape(b, c, h, w).float()
        input_d = input.permute(0, 2, 3, 1)
        image_d = torch.nn.functional.pad(input_d, (0, 0, pad, pad, pad, pad, 0, 0), mode='constant')  # N(H+2)(W+2)C
        for i in range(pad):
            j = i + 1
            image_d[:, 0 + i, :, :] = image_d[:, 1 + j, :, :]  # N(H+2)(W+2)C
            image_d[:, -1 - i, :, :] = image_d[:, -2 - j, :, :]  # N(H+2)(W+2)C
            image_d[:, :, 0 + i, :] = image_d[:, :, 1 + j, :]  # N(H+2)(W+2)C
            image_d[:, :, -1 - i, :] = image_d[:, :, -2 - j, :]  # N(H+2)(W+2)C

        image_d = image_d.permute(0, 3, 1, 2)
        unfolded = F.unfold(image_d, kernel_size=kernel_size)  # (b,c*l,h*w) l是滑动步数
        unfolded_re = unfolded.view(b, c, -1, h, w)  # (b,c,l,h,w)
        unfolded_re = unfolded_re.permute(2, 0, 1, 3, 4)  # (l,b,c,h,w)
        # 因为不需要和自己算，所以需要将自己置零
        # unfolded_re[kernel_size * kernel_size // 2, ...] = 0
        # 使用input_image 与 unfolded_re相乘就能得到它自己和邻居的乘积求和，当然，乘积只是举例
        # input_image(b,c,h,w)   unfolded_re(l,b,c,h,w)
        # 希望输出为(b,c,h,w)
        # result = torch.einsum('bchw,lbchw->bchw',[input,unfolded_re])
        if c == 1:
            return unfolded_re.long()
        return unfolded_re

    def pixel_contrast_loss(self, er_input, seg_label, seg_logit, gt_boundary_seg):
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
        # 首先提取出每个类各自的boundary (B,num_class,H,W)
        seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=256)[:, :, :, 0:self.num_classes].permute(0, 3,
                                                                                                                  1, 2)
        if self.same_class_extractor_weight.device != er_input.device:
            self.same_class_extractor_weight = self.same_class_extractor_weight.to(er_input.device)
            print("er move:", self.same_class_extractor_weight.device)
        if self.same_class_number_extractor_weight.device != er_input.device:
            self.same_class_number_extractor_weight = self.same_class_number_extractor_weight.to(er_input.device)
        # same_class_extractor是用来提取同一个类的邻居特征的
        same_class_extractor = NeighborExtractor5(self.extractor_channel)
        same_class_extractor = same_class_extractor.to(er_input.device)
        same_class_extractor.same_class_extractor.weight.data = self.same_class_extractor_weight
        # same_class_number_extractor是用来提取同一个类的邻居个数的
        same_class_number_extractor = NeighborExtractor5(1)
        same_class_number_extractor = same_class_number_extractor.to(er_input.device)
        same_class_number_extractor.same_class_extractor.weight.data = self.same_class_number_extractor_weight

        try:
            shown_class.remove(torch.tensor(255))
        except:
            pass
        # er_input = er_input.permute(0,2,3,1)
        contrast_loss_total = torch.tensor(0.0, device=er_input.device)
        cal_class_num = len(shown_class)
        for i in range(len(shown_class)):
            #----------------------------------------------------------------------------------
            # 获取当前类别的label掩膜，获取另外两个类别的label mask
            now_class_mask = seg_label_one_hot[:, shown_class[i].long(), :, :]
            pre_class_mask = seg_label_one_hot[:, shown_class[(i - 1) % len(shown_class)].long(), :, :]
            post_class_mask = seg_label_one_hot[:, shown_class[(i + 1) % len(shown_class)].long(), :, :]
            #----------------------------------------------------------------------------------

            #----------------------------------------------------------------------------------
            # 获取当前类别预测的掩膜以及其他类别的掩膜
            now_pred_class_mask = pred_label_one_hot[:, shown_class[i].long(), :, :]
            pre_pred_class_mask = pred_label_one_hot[:, shown_class[(i - 1) % len(shown_class)].long(), :, :]
            post_pred_class_mask = pred_label_one_hot[:, shown_class[(i + 1) % len(shown_class)].long(), :, :]
            #----------------------------------------------------------------------------------



            #----------------------------------------------------------------------------------
            # 下面是获得当前类的每个像素邻居中同属当前点的像素个数
            now_class_num_in_neigh = same_class_number_extractor(now_class_mask.unsqueeze(1).float())
            #----------------------------------------------------------------------------------

            #----------------------------------------------------------------------------------
            # 获取邻居中为正样本的邻居个数，也就是说即是当前像素的类别，又被预测正确的
            now_correct_class_num_in_neigh = same_class_number_extractor(
                (now_class_mask * now_pred_class_mask).unsqueeze(1).float())
            pre_correct_class_num_in_neigh = same_class_number_extractor(
                (pre_class_mask * pre_pred_class_mask).unsqueeze(1).float())
            post_correct_class_num_in_neigh = same_class_number_extractor(
                (post_class_mask * post_pred_class_mask).unsqueeze(1).float())
            #----------------------------------------------------------------------------------


            #--------------------- er loss的计算过程------------------------------------
            # 需要得到 可以参与er loss 计算的像素
            # 一个像素若要参与er loss计算，需要具备：
            # 1.邻居中具有同属当前类的像素:now_class_num_in_neigh
            # 2.当前像素要在边界上:edge_mask
            # 如果没有满足这些条件的像素 则直接跳过这个类的计算
            # now_class_mask

            # 这一句是将当前类别在边界gt中的位置给找出来
            pixel_cal_mask = (now_class_num_in_neigh.squeeze(1) >= 1) * (
                    edge_mask.bool() * now_class_mask.bool()).detach()
            # 这一句是将当前类别被预测对的位置在边界中找出来
            pixel_mse_cal_mask = (now_correct_class_num_in_neigh.squeeze(1) >= 1) * (
                    edge_mask.bool() * now_class_mask.bool() * now_pred_class_mask.bool()).detach()
            pre_pixel_cal_mask = (pre_correct_class_num_in_neigh.squeeze(1) >= 1) * (
                    edge_mask.bool() * pre_class_mask.bool() * pre_pred_class_mask.bool()).detach()
            post_pixel_cal_mask = (post_correct_class_num_in_neigh.squeeze(1) >= 1) * (
                    edge_mask.bool() * post_class_mask.bool() * post_pred_class_mask.bool()).detach()
            if pixel_cal_mask.sum() < 1 or pixel_mse_cal_mask.sum() < 1 or pre_pixel_cal_mask.sum() < 1 or post_pixel_cal_mask.sum() < 1:
                cal_class_num = cal_class_num - 1
                continue


            # 选择出参与erloss计算的像素的原始特征，哪些像素是要参与到计算中得呢？

            #------------ 计算contrast loss -------------------
            # 计算contrast loss应该要使用origin_mse_pixel_feat (1,256,n1,n2)
            # postive的特征也应该遵循同样的形状
            # 因此post_class_correct_forward_feat也应该形状与它一样(1,256,n1,n2)
            # 正样本是当前类别邻居的特征均值，这些均值后续一定会被分类正确

            # 新的解决方案：成功！
            # 直接使用原始er_input去获得每个元素周围的邻居，因为whole_neigh_feat是一个索引，所以可能会减少显存的开销
            whole_neigh_label = self.get_neigh(seg_label, kernel_size=5, pad=2).to(er_input.device) # (L,B,C,H,W)
            whole_neigh_pred = self.get_neigh(pred_label.unsqueeze(1), kernel_size=5, pad=2).to(er_input.device)
            whole_neigh_feat = self.get_neigh(er_input, kernel_size=5, pad=2).to(er_input.device) # (L,B,C,H,W)
            # 可以根据now_class_mask获得当前类别的坐标，从而直接取出它们的邻居和本身 (B,H,W)
            # whole_neigh_feat.permute(1,3,4,0,2)  (B,H,W,L,C)
            # .permute(1, 0) (num,L,C) 这里的num就是当前在boundary的类别邻居以及它本身在内的特征,但不知道哪些是正样本，哪些是负样本
            # 可以对该特征的gt也做同样的操作,这样就能拿到gt的unfold了,
            now_class_and_neigh_feat = whole_neigh_feat.permute(1,3,4,0,2)[pixel_cal_mask] # (num,L,C)
            now_class_and_neigh_label = whole_neigh_label.permute(1,3,4,0,2)[pixel_cal_mask] # (num,L,C)
            now_class_and_neigh_pred = whole_neigh_pred.permute(1,3,4,0,2)[pixel_cal_mask] # (num,L,C)
            # now_class_and_neigh_label的第一维度就是存储的类别标签
            # 计算对比损失的正样本是当前类别特征邻居的均值，负样本就是非当前类别的
            # 不妨将now_class_and_neigh_label转为one_hot (num,L,3),3代表的是类别
            # one_hot_now_class_and_neigh_label[:,:,0]，就拿到了第一个类别的标签


            one_hot_now_class_and_neigh_label = F.one_hot(now_class_and_neigh_label.squeeze(),num_classes = 3)
            one_hot_now_class_and_neigh_pred = F.one_hot(now_class_and_neigh_pred.squeeze(),num_classes = 3)

            one_hot_now_class_and_neigh_correct = one_hot_now_class_and_neigh_label * one_hot_now_class_and_neigh_pred
            # 这种情况下得到的则是128*25*256的矩阵
            now_class = now_class_and_neigh_feat * one_hot_now_class_and_neigh_correct[:, :, shown_class[i].long()].unsqueeze(-1)
            anchor = now_class[:,now_class.size()[1] // 2,:].unsqueeze(0)# 1*128*256
            now_class[:,now_class.size()[1] // 2,:] = 0 # 将自己置零
            # 计算当前类别被分类正确的邻居的feature均值
            contrast_positive = now_class.mean(dim = 1,keepdim=True).permute(1,0,2) # 1,128,256 代表128个像素，有一个256维度的中心
            # 因为负样本之间没有重叠，所以可以相加
            pre_class = (now_class_and_neigh_feat * one_hot_now_class_and_neigh_correct[:, :, shown_class[(i - 1) % len(shown_class)].long()].unsqueeze(-1)).permute(1,0,2)
            post_class =(now_class_and_neigh_feat * one_hot_now_class_and_neigh_correct[:, :,shown_class[(i + 1) % len(shown_class)].long()].unsqueeze(-1)).permute(1,0,2)
            contrast_negative = pre_class + post_class # (25,128,256)

            num_neigh = er_input.shape[0] // 2 -1
            padding = (num_neigh, 0)

            contrast_negative_unfold = torch.nn.functional.unfold(contrast_negative.unsqueeze(0),
                                                                  (2 * num_neigh + 1, contrast_negative.shape[2]),
                                                                  padding=padding). \
                view(contrast_negative.shape[0] * (2 * num_neigh + 1), contrast_negative.shape[1],
                     contrast_negative.shape[2])
            # (1,N,D),(1,N,D),(25,N,D)
            nce_loss = pixel_info_nce_loss(anchor,contrast_positive.detach(),contrast_negative_unfold.detach())
            contrast_loss_total = contrast_loss_total + nce_loss


        # 我的新对比损失：
        contrast_loss_total = contrast_loss_total / cal_class_num
        return contrast_loss_total

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

    def forward(self, outputs,gt_sem = None):
        gt_sem_boundary = self.gt2boundary(gt_sem.squeeze())
        pixel_contrast_loss = self.pixel_contrast_loss(
            outputs['out_features'],
            seg_label=gt_sem,
            seg_logit=outputs['out_classifier'],
            gt_boundary_seg=gt_sem_boundary,

        )
        return pixel_contrast_loss

if __name__ == '__main__':
    num_classes = 3
    ccbl = ContrastCrossPixelCorrect(num_classes)
    outputs = torch.zeros(4,512,512)
    seg_label = torch.zeros(4,512,512)
    seg_label[0,5:100] = 1.0
    seg_label[0,100:200] = 1.0
    seg_label[0,205:300] = 2.0
    seg_label[0,300:400] = 2.0
    loss = ccbl(outputs=outputs, gt_sem=seg_label)