import torch
import torch.nn.functional as F

def v1():
    # Input image (1 channel, 5x5 size)
    input_image = torch.arange(1, 151).reshape(2, 3, 5, 5).float()
    input_d = input_image.permute(0, 2, 3, 1)
    pad = 2
    image_d = torch.nn.functional.pad(input_d, (0, 0, pad, pad, pad, pad, 0, 0), mode='constant')  # N(H+2)(W+2)C
    for i in range(pad):
        j = i + 1
        image_d[:, 0 + i, :, :] = image_d[:, 1 + j, :, :]  # N(H+2)(W+2)C
        image_d[:, -1 - i, :, :] = image_d[:, -2 - j, :, :]  # N(H+2)(W+2)C
        image_d[:, :, 0 + i, :] = image_d[:, :, 1 + j, :]  # N(H+2)(W+2)C
        image_d[:, :, -1 - i, :] = image_d[:, :, -2 - j, :]  # N(H+2)(W+2)C

    image_d = image_d.permute(0, 3, 1, 2)

    # Apply unfold to input image with a kernel size of 5x5
    # 具体功能见./F_unfold.png
    unfolded = F.unfold(image_d, kernel_size=5)
    # 这里提取出来的数据就是之前数据的邻居的新tensor，那该如何使用呢
    # 原来的数据是BCHW，input (2,3,5,5),我们希望提取到每个像素的周围5*5的邻居出来，因此每个像素对应25个值，但是应用5*5卷积需要进行pad处理
    # pad后input_d (2,3,9,9)
    # input_d送入5*5的unfold，output:(2,75,25)
    # 2是batchsize
    # 75,25可以理解为 有25个邻居，每个邻居有3个通道的特征,所以是75=25*3

    # (2,3,25,5,5)
    # 表示的意思是batchsize为2
    # 3个通道
    # 每个元素有25个邻居
    # 原始图是5*5，因此有25个元素，这25个元素都有自己的邻居

    # 取第一个样本的第一个像素的全部特征
    # x[0,:,:,0,0]

    unfolded_re = unfolded.view(2, 3, 25, 25)
    print(input_image.shape)
    print(unfolded.shape)
    print(unfolded_re.shape)
    # print(input_image)
    # print(unfolded)
    # print("Input Image:")
    # print(input_image.size())
    # print(image_d.size())
    # print("\nUnfolded and Reshaped Output (9 channels):")
    # print(output_reshaped.size())

def get_neighv2(input,kernel_size = 3,pad = 1):
    b,c,h,w = input.size()
    input = torch.arange(1, b*c*h*w + 1).reshape(b,c,h,w).float()
    input_d = input.permute(0, 2, 3, 1)
    image_d = torch.nn.functional.pad(input_d, (0, 0, pad, pad, pad, pad, 0, 0), mode='constant')  # N(H+2)(W+2)C
    for i in range(pad):
        j = i + 1
        image_d[:, 0 + i, :, :] = image_d[:, 1 + j, :, :]  # N(H+2)(W+2)C
        image_d[:, -1 - i, :, :] = image_d[:, -2 - j, :, :]  # N(H+2)(W+2)C
        image_d[:, :, 0 + i, :] = image_d[:, :, 1 + j, :]  # N(H+2)(W+2)C
        image_d[:, :, -1 - i, :] = image_d[:, :, -2 - j, :]  # N(H+2)(W+2)C

    image_d = image_d.permute(0, 3, 1, 2)
    unfolded = F.unfold(image_d, kernel_size=kernel_size) #(b,c*l,h*w) l是滑动步数
    unfolded_re = unfolded.view(b,c,-1,h,w) #(b,c,l,h,w)
    unfolded_re = unfolded_re.permute(2,0,1,3,4)#(l,b,c,h,w)
    # 因为不需要和自己算，所以需要将自己置零
    unfolded_re[kernel_size*kernel_size//2,...] = 0
    # 使用input_image 与 unfolded_re相乘就能得到它自己和邻居的乘积求和，当然，乘积只是举例
    # input_image(b,c,h,w)   unfolded_re(l,b,c,h,w)
    # 希望输出为(b,c,h,w)
    # result = torch.einsum('bchw,lbchw->bchw',[input,unfolded_re])
    return unfolded_re

if __name__ == '__main__':
    """
    挑选邻居的条件：
    1、与当前类别相同的提取出来
    2、与当前类别相同且分类成功的
    3、与当前类别不同的，提取出来做负样本
    """
    input_image = torch.randn(4,256,64,64).float()

    # now_feat = er_input * now_class_mask.unsqueeze(1)#:仅获得当前类别的特征
    # now_correct_feat = er_input * (now_class_mask * now_pred_class_mask).unsqueeze(1)#:仅获得当前类别的特征里面在当前状态下被分类成功像素点的特征
    # pre_feat = er_input * pre_class_mask.unsqueeze(1)#:仅获得前一类别的特征
    # pre_correct_feat = er_input * (pre_class_mask * pre_pred_class_mask).unsqueeze(1)
    # post_feat = er_input * post_class_mask.unsqueeze(1)#:仅获得后一类别的特征
    # post_correct_feat = er_input * (post_class_mask * post_pred_class_mask).unsqueeze(1)

    ####### 如果要考虑分类是否正确，也可以选择不同的feat
    # now_and_pre_feat = '' #now_feat + pre_feat #:获得当前特征和之前一个类别的特征
    # now_and_post_feat = '' #now_feat + pre_feat #:获得当前特征和后一个类别的特征
    # # 返回的就是当前特征像素点周围的pre 类别的特征，是负样本 lbchw
    # pre_feats_neigh_for_now = get_neighv2(get_neighv2(now_and_pre_feat,kernel_size=5,pad = 2))
    # # 返回的就是当前特征像素点周围的post 类别的特征，是负样本 lbchw
    # post_feats_neigh_for_now = get_neighv2(get_neighv2(now_and_post_feat,kernel_size=5,pad = 2))

    neighs = get_neighv2(input_image,kernel_size=5,pad = 2)
    neighs_shaped = neighs.reshape(neighs.size(0), -1) #(l, b*c*h*w)
    input_image_reshaped = input_image.reshape(-1) # (b*c*h*w)
    negative_cosine_similarity = F.cosine_similarity(input_image_reshaped.unsqueeze(0), neighs_shaped, dim=1)
    sum_negative_cosine_similarity = negative_cosine_similarity.sum(dim=0)

    new_cosine_similarity = F.cosine_similarity(input_image.unsqueeze(0), neighs, dim=0)
    sum_now_negative_cosine_similarity = new_cosine_similarity.sum(dim=0)

    # input_image_shaped = input_image.view(1, -1, input_image.size(2), input_image.size(3)) # 1,-1,h,w
    # cosine_similarity = F.cosine_similarity(input_image_shaped,neighs_shaped,dim=1)

    print(sum_negative_cosine_similarity)
    print(sum_now_negative_cosine_similarity)
