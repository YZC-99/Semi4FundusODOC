import torch
import torch.nn.functional as F

def get_neigh(self, input, kernel_size=3, pad=1):
    b, c, h, w = input.size()
    input = torch.arange(1, b * c * h * w + 1).reshape(b, c, h, w).float()
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
    unfolded_re[kernel_size * kernel_size // 2, ...] = 0
    # 使用input_image 与 unfolded_re相乘就能得到它自己和邻居的乘积求和，当然，乘积只是举例
    # input_image(b,c,h,w)   unfolded_re(l,b,c,h,w)
    # 希望输出为(b,c,h,w)
    # result = torch.einsum('bchw,lbchw->bchw',[input,unfolded_re])
    return unfolded_re

def gt2boundary(gt, ignore_label=-1,boundary_width = 5):  # gt NHW
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