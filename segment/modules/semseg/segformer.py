# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment.modules.semseg.nn import Attention,CrissCrossAttention,CoordAtt,CBAMBlock
from segment.modules.backbone.resnet import resnet18,resnet34, resnet50, resnet101
from segment.modules.backbone.mit import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1,attention='subv1'):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.attention = attention

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        if self.attention == 'subv1':
            # 我自己加的
            self.criss_cross_attention_sub1 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub2 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub1_cross_sub2 = CrissCrossAttention(embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif self.attention == 'subv1-1':
            # 我自己加的
            self.criss_cross_attention_sub1 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub2 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub1_cross_sub2_R1 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub1_cross_sub2_R2 = CrissCrossAttention(embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif self.attention == 'subv1-2':
            self.criss_cross_attention_sub1 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub2 = CrissCrossAttention(embedding_dim)
            self.cca_sub1 = CrissCrossAttention(embedding_dim)
            self.cca_sub2 = CrissCrossAttention(embedding_dim)
            self.cca_sub1_sub2 = CrissCrossAttention(embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif self.attention == 'subv2':
            # 我自己加的
            self.criss_cross_attention_sub1 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub2 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub3 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub4 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub1_cross_sub2 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub3_cross_sub4 = CrissCrossAttention(embedding_dim)


            self.linear_fuse = ConvModule(
                c1=embedding_dim*6,
                c2=embedding_dim,
                k=1,
            )
        elif self.attention == 'subv3':
            # 我自己加的
            self.criss_cross_attention_sub1 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub2 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub3 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub4 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub1_cross_sub2 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub3_cross_sub4 = CrissCrossAttention(embedding_dim)

            self.criss_cross_attention_sub1_2_cross_sub3_4 = CrissCrossAttention(embedding_dim)


            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif self.attention == 'subv4':
            # 我自己加的
            self.criss_cross_attention_sub1 = CrissCrossAttention(embedding_dim)
            self.criss_cross_attention_sub2 = CrissCrossAttention(embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*6,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'ccav1':
            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.cca3 = CrissCrossAttention(embedding_dim)
            self.cca4 = CrissCrossAttention(embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim * 4,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv1':
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 6,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv2':
            self.linear_sub_fuse = ConvModule(
                c1=embedding_dim * 2,
                c2=embedding_dim,
                k=1,
            )
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv3':
            self.linear_sub_fuse = ConvModule(
                c1=embedding_dim * 2,
                c2=embedding_dim,
                k=1,
            )
            self.linear_sub_fuse_cca = CrissCrossAttention(embedding_dim)
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv3-1':
            self.linear_sub_fuse = ConvModule(
                c1=embedding_dim * 2,
                c2=embedding_dim,
                k=1,
            )
            self.linear_sub_fuse_cbam = CBAMBlock(channel=embedding_dim,
                                                 reduction=16,
                                                 kernel_size=65)
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv4':
            self.linear_sub_fuse1 = ConvModule(
                c1=embedding_dim * 2,
                c2=embedding_dim,
                k=1,
            )
            self.linear_sub_fuse2 = ConvModule(
                c1=embedding_dim * 2,
                c2=embedding_dim,
                k=1,
            )
            self.linear_sub_fuse_cca1 = CrissCrossAttention(embedding_dim)
            self.linear_sub_fuse_cca2 = CrissCrossAttention(embedding_dim)
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 6,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv5':
            self.linear_sub_fuse1 = ConvModule(
                c1=embedding_dim * 2,
                c2=embedding_dim,
                k=1,
            )
            self.linear_sub_fuse2 = ConvModule(
                c1=embedding_dim * 2,
                c2=embedding_dim,
                k=1,
            )
            self.linear_sub_fuse_cca1 = CrissCrossAttention(embedding_dim)
            self.linear_sub_fuse_cca2 = CrissCrossAttention(embedding_dim)

            self.cca_fuse = ConvModule(
                c1=embedding_dim * 2,
                c2=embedding_dim,
                k=1,
            )

            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv6':
            self.cca_fuse = ConvModule(
                c1=embedding_dim * 6,
                c2=embedding_dim,
                k=1,
            )

            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv7':
            self.sub_cca1 = CrissCrossAttention(embedding_dim)
            self.sub_cca2 = CrissCrossAttention(embedding_dim)
            self.sub_cca3 = CrissCrossAttention(embedding_dim)
            self.sub_cca4 = CrissCrossAttention(embedding_dim)
            self.sub_cca5 = CrissCrossAttention(embedding_dim)
            self.sub_cca6 = CrissCrossAttention(embedding_dim)
            self.cca_fuse = ConvModule(
                c1=embedding_dim * 6,
                c2=embedding_dim,
                k=1,
            )

            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_addv8':
            self.sub_cca1 = CrissCrossAttention(embedding_dim*3)
            self.sub_cca2 = CrissCrossAttention(embedding_dim*2)
            self.cca_fuse = ConvModule(
                c1=embedding_dim * 6,
                c2=embedding_dim,
                k=1,
            )
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'multi_addv1':
            self.sub_cca1 = CrissCrossAttention(embedding_dim*3)
            self.sub_cca2 = CrissCrossAttention(embedding_dim*2)
            self.cca_fuse = ConvModule(
                c1=embedding_dim * 6,
                c2=embedding_dim,
                k=1,
            )
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'add_addv1':
            self.sub_cca = CrissCrossAttention(embedding_dim*4)
            self.cca_fuse = ConvModule(
                c1=embedding_dim * 4,
                c2=embedding_dim,
                k=1,
            )
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'sub_or_multi_addv1':
            self.sub_cca1 = CrissCrossAttention(embedding_dim*3)
            self.sub_cca2 = CrissCrossAttention(embedding_dim*2)
            self.cca_fuse = ConvModule(
                c1=embedding_dim * 6,
                c2=embedding_dim,
                k=1,
            )
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv1':
            self.lateral_c1 = ConvModule(c1_in_channels,768)
            self.lateral_c2 = ConvModule(c2_in_channels,768)
            self.lateral_c3 = ConvModule(c3_in_channels,768)
            self.lateral_c4 = ConvModule(c4_in_channels,768)
            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.cca3 = CrissCrossAttention(embedding_dim)
            self.FFN_multi_level = ConvModule(embedding_dim,768)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv2':
            self.lateral_c1 = ConvModule(c1_in_channels,128)
            self.lateral_c2 = ConvModule(c2_in_channels,128)
            self.lateral_c3 = ConvModule(c3_in_channels,256)
            self.lateral_c4 = ConvModule(c4_in_channels,384)
            self.cca1 = CrissCrossAttention(128)
            self.cca2 = CrissCrossAttention(256)
            self.cca3 = CrissCrossAttention(384)
            self.FFN_multi_level = ConvModule(384,768)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )

        elif attention == 'backbone_multi-levelv3':
            self.lateral_c1 = ConvModule(c1_in_channels,768)
            self.lateral_c2 = ConvModule(c2_in_channels,768)
            self.lateral_c3 = ConvModule(c3_in_channels,768)
            self.lateral_c4 = ConvModule(c4_in_channels,768)
            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.cca3 = CrissCrossAttention(embedding_dim)

            self.multi_linear_c4 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
            self.multi_linear_c3 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
            self.multi_linear_c2 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
            self.multi_linear_c1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*4,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv4':
            self.lateral_c1 = ConvModule(c1_in_channels,768)
            self.lateral_c2 = ConvModule(c2_in_channels,768)
            self.lateral_c3 = ConvModule(c3_in_channels,768)
            self.lateral_c4 = ConvModule(c4_in_channels,768)
            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.cca3 = CrissCrossAttention(embedding_dim)

            self.multi_linear_c4 = ConvModule(embedding_dim,embedding_dim)
            self.multi_linear_c3 = ConvModule(embedding_dim,embedding_dim)
            self.multi_linear_c2 = ConvModule(embedding_dim,embedding_dim)
            self.multi_linear_c1 = ConvModule(embedding_dim,embedding_dim)


            self.linear_fuse = ConvModule(
                c1=embedding_dim*4,
                c2=embedding_dim,
                k=1,
            )

        elif attention == 'backbone_multi-levelv5':
            # self.lateral_c1 = ConvModule(c1_in_channels,768)
            # self.lateral_c2 = ConvModule(c2_in_channels,768)
            # self.lateral_c3 = ConvModule(c3_in_channels,768)
            # self.lateral_c4 = ConvModule(c4_in_channels,768)
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.cca1 = CrissCrossAttention(c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels,c1_in_channels)
            self.cca2 = CrissCrossAttention(c1_in_channels)
            self.ffn3 = ConvModule(c1_in_channels+c1_in_channels,embedding_dim)
            self.cca3 = CrissCrossAttention(embedding_dim)

        elif attention == 'backbone_multi-levelv6':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels,c1_in_channels)
            self.ffn3 = ConvModule(c1_in_channels+c1_in_channels,embedding_dim)
        elif attention == 'backbone_multi-levelv7':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels,c1_in_channels)
            self.ffn3 = ConvModule(c1_in_channels+c1_in_channels,embedding_dim)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c2_in_channels + c1_in_channels + embedding_dim,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-i':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,256)
            self.ffn2 = ConvModule(c2_in_channels+256,192)
            self.ffn3 = ConvModule(c1_in_channels+192,128)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + 256 + 192 + 128,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels,c1_in_channels)
            self.ffn3 = ConvModule(c1_in_channels+c1_in_channels,32)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c2_in_channels + c1_in_channels + 32,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c3_in_channels + c2_in_channels + c1_in_channels,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-1':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)
            self.cca1 = CrissCrossAttention(c1_in_channels)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c3_in_channels + c2_in_channels + c1_in_channels,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-2':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,embedding_dim)
            self.ffn2 = ConvModule(embedding_dim+c2_in_channels,embedding_dim)
            self.ffn3 = ConvModule(embedding_dim+c1_in_channels,embedding_dim)

            self.linear_fuse = ConvModule(
                c1=c4_in_channels + embedding_dim + embedding_dim + embedding_dim,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-3':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)

            self.linear_c1 = MLP(input_dim=c1_in_channels,embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels,embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels,embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=c4_in_channels,embed_dim=embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*4,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-4':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)

            self.cca1 = CrissCrossAttention(c1_in_channels)
            self.cca2 = CrissCrossAttention(c2_in_channels)
            self.cca3 = CrissCrossAttention(c3_in_channels)
            self.cca4 = CrissCrossAttention(c4_in_channels)

            self.linear_c1 = MLP(input_dim=c1_in_channels,embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels,embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels,embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=c4_in_channels,embed_dim=embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*4,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-5':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)

            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.cca3 = CrissCrossAttention(embedding_dim)

            self.linear_c1 = MLP(input_dim=c1_in_channels,embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels,embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels,embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=c4_in_channels,embed_dim=embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-6':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)


            self.linear_c1 = MLP(input_dim=c1_in_channels,embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels,embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels,embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=c4_in_channels,embed_dim=embedding_dim)

            self.ffn_sub = ConvModule(embedding_dim * 2,embedding_dim)
            self.cca1 = CrissCrossAttention(embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-7':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)

            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.cca3 = CrissCrossAttention(embedding_dim)
            self.cca4 = CrissCrossAttention(embedding_dim)

            self.linear_c1 = MLP(input_dim=c1_in_channels,embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels,embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels,embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=c4_in_channels,embed_dim=embedding_dim)

            self.linear_fuse = ConvModule(
                c1=embedding_dim*4,
                c2=embedding_dim,
                k=1,
            )

        elif attention == 'backbone_multi-levelv7-iii':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels+c4_in_channels,256)
            self.ffn3 = ConvModule(256+c2_in_channels+c1_in_channels+c4_in_channels,embedding_dim)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c2_in_channels + 256 + embedding_dim,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-iv':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels+c4_in_channels,256)
            self.ffn3 = ConvModule(256+c2_in_channels+c1_in_channels,384)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c2_in_channels + 256 + 384,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-1':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels,c1_in_channels)
            self.ffn3 = ConvModule(c1_in_channels+c1_in_channels,embedding_dim)
            self.ffn_sub = ConvModule(c4_in_channels,c4_in_channels)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c2_in_channels + c1_in_channels + embedding_dim + c4_in_channels,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-2':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels,c1_in_channels)
            self.ffn3 = ConvModule(c1_in_channels+c1_in_channels,embedding_dim)
            self.cca_sub = CrissCrossAttention(c4_in_channels)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c2_in_channels + c1_in_channels + embedding_dim + c4_in_channels,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-3':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels,c1_in_channels)
            self.ffn3 = ConvModule(c1_in_channels+c1_in_channels,embedding_dim)
            self.cca_sub = CrissCrossAttention(c4_in_channels)
            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c2_in_channels + c1_in_channels + embedding_dim,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv8':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c2_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c2_in_channels,c1_in_channels)
            self.ffn3 = ConvModule(c1_in_channels+c1_in_channels,embedding_dim)

            self.cca1 = CrissCrossAttention(c2_in_channels)
            self.cca2 = CrissCrossAttention(c1_in_channels)
            self.cca3 = CrissCrossAttention(embedding_dim)
            self.cca4 = CrissCrossAttention(c4_in_channels)

            self.linear_fuse = ConvModule(
                c1=c4_in_channels + c2_in_channels + c1_in_channels + embedding_dim,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_subv1':
            self.lateral_c1 = ConvModule(c1_in_channels,768)
            self.lateral_c2 = ConvModule(c2_in_channels,768)
            self.lateral_c3 = ConvModule(c3_in_channels,768)
            self.lateral_c4 = ConvModule(c4_in_channels,768)
            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.linear_sub_fuse = MLP(input_dim=embedding_dim * 2, embed_dim=embedding_dim)
            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_subv2':
            self.lateral_c1 = ConvModule(c1_in_channels,768)
            self.lateral_c2 = ConvModule(c2_in_channels,768)
            self.lateral_c3 = ConvModule(c3_in_channels,768)
            self.lateral_c4 = ConvModule(c4_in_channels,768)
            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.linear_fuse = ConvModule(
                c1=embedding_dim*6,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_addv1':
            self.lateral_c1 = ConvModule(c1_in_channels,768)
            self.lateral_c2 = ConvModule(c2_in_channels,768)
            self.lateral_c3 = ConvModule(c3_in_channels,768)
            self.lateral_c4 = ConvModule(c4_in_channels,768)
            self.cca1 = CrissCrossAttention(embedding_dim)
            self.cca2 = CrissCrossAttention(embedding_dim)
            self.linear_fuse = ConvModule(
                c1=embedding_dim*6,
                c2=embedding_dim,
                k=1,
            )
        else:
            self.linear_fuse = ConvModule(
                c1=embedding_dim*4,
                c2=embedding_dim,
                k=1,
            )


        # self.classifier    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        #
        if self.attention == 'subv1':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub1_att = self.criss_cross_attention_sub1(sub1)
            sub2_att = self.criss_cross_attention_sub2(sub2)
            sub1_cross_sub2_att = self.criss_cross_attention_sub1_cross_sub2.cross_forward(sub1_att,sub2_att)
            _c = self.linear_fuse(torch.cat([sub1_cross_sub2_att,_c4, _c3, _c2, _c1], dim=1))
        elif self.attention == 'subv1-1':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub1_att = self.criss_cross_attention_sub1(sub1)
            sub2_att = self.criss_cross_attention_sub2(sub2)
            sub1_cross_sub2_att_R1 = self.criss_cross_attention_sub1_cross_sub2_R1.cross_forward(sub1_att,sub2_att)
            sub1_cross_sub2_att_R2 = self.criss_cross_attention_sub1_cross_sub2_R2(sub1_cross_sub2_att_R1)
            _c = self.linear_fuse(torch.cat([sub1_cross_sub2_att_R2,_c4, _c3, _c2, _c1], dim=1))
        elif self.attention == 'subv1-2':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub1_att = self.criss_cross_attention_sub1(sub1)
            sub2_att = self.criss_cross_attention_sub2(sub2)
            sub1_att_cca = self.cca_sub1(sub1_att)
            sub2_att_cca = self.cca_sub2(sub2_att)
            sub1_sub2_att_cca = self.cca_sub1_sub2.cross_forward(sub1_att_cca,sub2_att_cca)
            _c = self.linear_fuse(torch.cat([sub1_sub2_att_cca, _c4, _c3, _c2, _c1], dim=1))
        elif self.attention == 'subv2':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub1_att = self.criss_cross_attention_sub1(sub1)
            sub2_att = self.criss_cross_attention_sub2(sub2)
            sub1_cross_sub2_att = self.criss_cross_attention_sub1_cross_sub2.cross_forward(sub1_att,sub2_att)

            sub3 = _c2 - _c3
            sub4 = _c2 - _c4
            sub3_att = self.criss_cross_attention_sub3(sub3)
            sub4_att = self.criss_cross_attention_sub4(sub4)
            sub3_cross_sub4_att = self.criss_cross_attention_sub3_cross_sub4.cross_forward(sub3_att, sub4_att)
            _c = self.linear_fuse(torch.cat([sub1_cross_sub2_att,sub3_cross_sub4_att, _c4, _c3, _c2, _c1], dim=1))
        elif self.attention == 'subv3':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub1_att = self.criss_cross_attention_sub1(sub1)
            sub2_att = self.criss_cross_attention_sub2(sub2)
            sub1_cross_sub2_att = self.criss_cross_attention_sub1_cross_sub2.cross_forward(sub1_att, sub2_att)

            sub3 = _c2 - _c3
            sub4 = _c2 - _c4
            sub3_att = self.criss_cross_attention_sub3(sub3)
            sub4_att = self.criss_cross_attention_sub4(sub4)
            sub3_cross_sub4_att = self.criss_cross_attention_sub3_cross_sub4.cross_forward(sub3_att, sub4_att)

            sub3_cross_sub4_att = self.criss_cross_attention_sub1_2_cross_sub3_4.cross_forward(sub1_cross_sub2_att, sub3_cross_sub4_att)

            _c = self.linear_fuse(torch.cat([sub3_cross_sub4_att, _c4, _c3, _c2, _c1], dim=1))

        elif self.attention == 'subv4':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub1_att = self.criss_cross_attention_sub1(sub1)
            sub2_att = self.criss_cross_attention_sub2(sub2)
            _c = self.linear_fuse(torch.cat([sub1_att,sub2_att, _c4, _c3, _c2, _c1], dim=1))
        elif self.attention == 'ccav1':
            _c1 = self.cca1(_c1)
            _c2 = self.cca1(_c2)
            _c3 = self.cca1(_c3)
            _c4 = self.cca1(_c4)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        elif self.attention == 'sub_addv1':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1,sub1,sub2], dim=1))
        elif self.attention == 'sub_addv2':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            _sub = self.linear_sub_fuse(torch.cat([sub1,sub2],dim=1))
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, _sub], dim=1))
        elif self.attention == 'sub_addv3':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            _sub = self.linear_sub_fuse(torch.cat([sub1,sub2],dim=1))
            _sub = self.linear_sub_fuse_cca(_sub)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, _sub], dim=1))
        elif self.attention == 'sub_addv3-1':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            _sub = self.linear_sub_fuse(torch.cat([sub1,sub2],dim=1))
            _sub = self.linear_sub_fuse_cbam(_sub)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, _sub], dim=1))
        elif self.attention == 'sub_addv4':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub3 = _c1 - _c3
            sub4 = _c2 - _c4
            _sub1 = self.linear_sub_fuse1(torch.cat([sub1,sub2],dim=1))
            _sub1 = self.linear_sub_fuse_cca1(_sub1)
            _sub2 = self.linear_sub_fuse1(torch.cat([sub3,sub4],dim=1))
            _sub2 = self.linear_sub_fuse_cca1(_sub2)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, _sub1,_sub2], dim=1))
        elif self.attention == 'sub_addv5':
            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub3 = _c1 - _c3
            sub4 = _c2 - _c4
            _sub1 = self.linear_sub_fuse1(torch.cat([sub1,sub2],dim=1))
            _sub1 = self.linear_sub_fuse_cca1(_sub1)
            _sub2 = self.linear_sub_fuse1(torch.cat([sub3,sub4],dim=1))
            _sub2 = self.linear_sub_fuse_cca1(_sub2)
            cca_fuse = self.cca_fuse(torch.cat([_sub1,_sub2], dim=1))
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1,cca_fuse], dim=1))
        elif self.attention == 'sub_addv6':
            sub1 = _c1 - _c2
            sub2 = _c1 - _c3
            sub3 = _c1 - _c4
            sub4 = _c2 - _c3
            sub5 = _c2 - _c4
            sub6 = _c3 - _c4
            cca_fuse = self.cca_fuse(torch.cat([sub1,sub2,sub3,sub4,sub5,sub6], dim=1))
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, cca_fuse], dim=1))
        elif self.attention == 'sub_addv7':
            sub1 = _c1 - _c2
            sub2 = _c1 - _c3
            sub3 = _c1 - _c4
            sub4 = _c2 - _c3
            sub5 = _c2 - _c4
            sub6 = _c3 - _c4
            sub_cca1 = self.sub_cca1(sub1)
            sub_cca2 = self.sub_cca1(sub2)
            sub_cca3 = self.sub_cca1(sub3)
            sub_cca4 = self.sub_cca1(sub4)
            sub_cca5 = self.sub_cca1(sub5)
            sub_cca6 = self.sub_cca1(sub6)
            cca_fuse = self.cca_fuse(torch.cat([sub_cca1,sub_cca2,sub_cca3,sub_cca4,sub_cca5,sub_cca6], dim=1))
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, cca_fuse], dim=1))
        elif self.attention == 'sub_addv8':
            sub1 = _c1 - _c2
            sub2 = _c1 - _c3
            sub3 = _c1 - _c4
            sub4 = _c2 - _c3
            sub5 = _c2 - _c4
            sub6 = _c3 - _c4

            sub_cca1 = self.sub_cca1(torch.cat([sub1,sub2,sub3], dim=1))
            sub_cca2 = self.sub_cca2(torch.cat([sub4,sub5], dim=1))
            cca_fuse = self.cca_fuse(torch.cat([sub_cca1, sub_cca2,sub6], dim=1))
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, cca_fuse], dim=1))
        elif self.attention == 'multi_addv1':
            sub1 = _c1 * _c2
            sub2 = _c1 * _c3
            sub3 = _c1 * _c4
            sub4 = _c2 * _c3
            sub5 = _c2 * _c4
            sub6 = _c3 * _c4

            sub_cca1 = self.sub_cca1(torch.cat([sub1, sub2, sub3], dim=1))
            sub_cca2 = self.sub_cca2(torch.cat([sub4, sub5], dim=1))
            cca_fuse = self.cca_fuse(torch.cat([sub_cca1, sub_cca2, sub6], dim=1))
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, cca_fuse], dim=1))
        elif self.attention == 'add_addv1':
            sub_cca = self.sub_cca(torch.cat([_c1,_c2,_c3,_c4], dim=1))
            cca_fuse = self.cca_fuse(sub_cca)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, cca_fuse], dim=1))
        elif self.attention == 'sub_or_multi_addv1':
            p = 0.5
            sub1 = _c1 * _c2 if random.random() < p else _c1 - _c2
            sub2 = _c1 * _c3 if random.random() < p else _c1 - _c3
            sub3 = _c1 * _c4 if random.random() < p else _c1 - _c4
            sub4 = _c2 * _c3 if random.random() < p else _c2 - _c3
            sub5 = _c2 * _c4 if random.random() < p else _c2 - _c4
            sub6 = _c3 * _c4 if random.random() < p else _c3 - _c4

            sub_cca1 = self.sub_cca1(torch.cat([sub1, sub2, sub3], dim=1))
            sub_cca2 = self.sub_cca2(torch.cat([sub4, sub5], dim=1))
            cca_fuse = self.cca_fuse(torch.cat([sub_cca1, sub_cca2, sub6], dim=1))
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, cca_fuse], dim=1))
        elif self.attention == 'backbone_multi-levelv1':
            # 先统一通道
            lateral_c1 = self.lateral_c1(c1)
            lateral_c2 = self.lateral_c2(c2)
            lateral_c3 = self.lateral_c3(c3)
            lateral_c4 = self.lateral_c4(c4)

            # 全部上采样到128*128
            lateral_c2 = F.interpolate(lateral_c2, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(lateral_c3, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(lateral_c4, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)

            # 做cross criss attention
            cca1 = self.cca1.cross_forward(lateral_c1,lateral_c2)
            cca2 = self.cca2.cross_forward(cca1,lateral_c3)
            cca3 = self.cca3.cross_forward(cca2,lateral_c4)
            ffn_result = self.FFN_multi_level(cca3)

            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, ffn_result], dim=1))

        elif self.attention == 'backbone_multi-levelv2':
            lateral_c1 = self.lateral_c1(c1) # 128
            lateral_c2 = self.lateral_c2(c2) # 128
            lateral_c3 = self.lateral_c3(c3) # 256
            lateral_c4 = self.lateral_c4(c4) # 384

            # 全部上采样到128*128
            lateral_c2 = F.interpolate(lateral_c2, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(lateral_c3, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(lateral_c4, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)

            cca1 = self.cca1.cross_forward(lateral_c1,lateral_c2)
            cca2 = self.cca2.cross_forward(torch.cat([cca1,lateral_c1],dim=1),lateral_c3)
            cca3 = self.cca3.cross_forward(torch.cat([cca2,lateral_c2],dim=1),lateral_c4)

            ffn_result = self.FFN_multi_level(cca3)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, ffn_result], dim=1))

        elif self.attention == 'backbone_multi-levelv3':
            # 先统一通道
            lateral_c1 = self.lateral_c1(c1)
            lateral_c2 = self.lateral_c2(c2)
            lateral_c3 = self.lateral_c3(c3)
            lateral_c4 = self.lateral_c4(c4)
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(lateral_c2, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(lateral_c3, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(lateral_c4, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            # 做cross criss attention
            cca1 = self.cca1.cross_forward(lateral_c4,lateral_c3)
            cca2 = self.cca2.cross_forward(cca1,lateral_c2)
            cca3 = self.cca3.cross_forward(cca2,lateral_c1)

            n, _, h, w = cca3.shape
            _c1 = self.multi_linear_c1(cca1).permute(0,2,1).reshape(n, -1, cca1.shape[2], cca1.shape[3])
            _c2 = self.multi_linear_c1(cca2).permute(0,2,1).reshape(n, -1, cca2.shape[2], cca2.shape[3])
            _c3 = self.multi_linear_c1(cca3).permute(0,2,1).reshape(n, -1, cca3.shape[2], cca3.shape[3])
            _c4 = self.multi_linear_c1(lateral_c4).permute(0,2,1).reshape(n, -1, lateral_c4.shape[2], lateral_c4.shape[3])

            _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))

        elif self.attention == 'backbone_multi-levelv4':
            # 先统一通道
            lateral_c1 = self.lateral_c1(c1)
            lateral_c2 = self.lateral_c2(c2)
            lateral_c3 = self.lateral_c3(c3)
            lateral_c4 = self.lateral_c4(c4)
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(lateral_c2, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(lateral_c3, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(lateral_c4, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            # 做cross criss attention
            cca1 = self.cca1.cross_forward(lateral_c4,lateral_c3)
            cca2 = self.cca2.cross_forward(cca1,lateral_c2)
            cca3 = self.cca3.cross_forward(cca2,lateral_c1)
            _c1 = self.multi_linear_c1(cca1)
            _c2 = self.multi_linear_c1(cca2)
            _c3 = self.multi_linear_c1(cca3)
            _c4 = self.multi_linear_c1(lateral_c4)
            _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))

        elif self.attention == 'backbone_multi-levelv5':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out1_cca = self.cca1(out1)
            out2 = self.ffn2(torch.cat([out1_cca,lateral_c2],dim=1))
            out2_cca = self.cca2(out2)
            out3 = self.ffn3(torch.cat([out2_cca,c1],dim=1))
            _c = self.cca3(out3)
        elif self.attention == 'backbone_multi-levelv6':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            _c = self.ffn3(torch.cat([out2,c1],dim=1))
        elif self.attention == 'backbone_multi-levelv7':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            _c = self.linear_fuse(torch.cat([lateral_c4,out1,out2,out3],dim=1))
        elif self.attention == 'backbone_multi-levelv7-i':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            _c = self.linear_fuse(torch.cat([lateral_c4,out1,out2,out3],dim=1))
        elif self.attention == 'backbone_multi-levelv7-ii':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            _c = self.linear_fuse(torch.cat([lateral_c4,out1,out2,out3],dim=1))
        elif self.attention == 'backbone_multi-levelv7-ii-1':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            _c = self.linear_fuse(torch.cat([lateral_c4,out1,out2,out3],dim=1))

        elif self.attention == 'backbone_multi-levelv7-ii-1-1':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            out3 = self.cca1(out3)
            _c = self.linear_fuse(torch.cat([lateral_c4,out1,out2,out3],dim=1))

        elif self.attention == 'backbone_multi-levelv7-ii-1-2':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            _c = self.linear_fuse(torch.cat([lateral_c4,out1,out2,out3],dim=1))

        elif self.attention == 'backbone_multi-levelv7-ii-1-3':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            _c4 = self.linear_c4(lateral_c4).permute(0,2,1).reshape(n, -1, lateral_c4.shape[2], lateral_c4.shape[3])
            _c3 = self.linear_c3(out1).permute(0,2,1).reshape(n, -1, out1.shape[2], out1.shape[3])
            _c2 = self.linear_c2(out2).permute(0,2,1).reshape(n, -1, out2.shape[2], out2.shape[3])
            _c1 = self.linear_c1(out3).permute(0,2,1).reshape(n, -1, out3.shape[2], out3.shape[3])

            _c = self.linear_fuse(torch.cat([_c1,_c2,_c3,_c4],dim=1))

        elif self.attention == 'backbone_multi-levelv7-ii-1-4':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            lateral_c4 = self.cca4(lateral_c4)
            out1 = self.cca3(out1)
            out2 = self.cca2(out2)
            out3 = self.cca1(out3)

            _c4 = self.linear_c4(lateral_c4).permute(0,2,1).reshape(n, -1, lateral_c4.shape[2], lateral_c4.shape[3])
            _c3 = self.linear_c3(out1).permute(0,2,1).reshape(n, -1, out1.shape[2], out1.shape[3])
            _c2 = self.linear_c2(out2).permute(0,2,1).reshape(n, -1, out2.shape[2], out2.shape[3])
            _c1 = self.linear_c1(out3).permute(0,2,1).reshape(n, -1, out3.shape[2], out3.shape[3])

            _c = self.linear_fuse(torch.cat([_c1,_c2,_c3,_c4],dim=1))

        elif self.attention == 'backbone_multi-levelv7-ii-1-5':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))


            _c4 = self.linear_c4(lateral_c4).permute(0,2,1).reshape(n, -1, lateral_c4.shape[2], lateral_c4.shape[3])
            _c3 = self.linear_c3(out1).permute(0,2,1).reshape(n, -1, out1.shape[2], out1.shape[3])
            _c2 = self.linear_c2(out2).permute(0,2,1).reshape(n, -1, out2.shape[2], out2.shape[3])
            _c1 = self.linear_c1(out3).permute(0,2,1).reshape(n, -1, out3.shape[2], out3.shape[3])

            cca1 = self.cca1.cross_forward(_c3,_c4)
            cca2 = self.cca2.cross_forward(_c2,cca1)
            cca3 = self.cca3.cross_forward(_c1,cca2)

            _c = self.linear_fuse(torch.cat([_c1,_c2,_c3,_c4,cca3],dim=1))

        elif self.attention == 'backbone_multi-levelv7-ii-1-6':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))

            _c4 = self.linear_c4(lateral_c4).permute(0,2,1).reshape(n, -1, lateral_c4.shape[2], lateral_c4.shape[3])
            _c3 = self.linear_c3(out1).permute(0,2,1).reshape(n, -1, out1.shape[2], out1.shape[3])
            _c2 = self.linear_c2(out2).permute(0,2,1).reshape(n, -1, out2.shape[2], out2.shape[3])
            _c1 = self.linear_c1(out3).permute(0,2,1).reshape(n, -1, out3.shape[2], out3.shape[3])

            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub = self.ffn_sub(torch.cat([sub1,sub2],dim=1))
            sub = self.cca1(sub)

            _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4, sub], dim=1))
        elif self.attention == 'backbone_multi-levelv7-ii-1-7':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            _c4 = self.linear_c4(lateral_c4).permute(0,2,1).reshape(n, -1, lateral_c4.shape[2], lateral_c4.shape[3])
            _c3 = self.linear_c3(out1).permute(0,2,1).reshape(n, -1, out1.shape[2], out1.shape[3])
            _c2 = self.linear_c2(out2).permute(0,2,1).reshape(n, -1, out2.shape[2], out2.shape[3])
            _c1 = self.linear_c1(out3).permute(0,2,1).reshape(n, -1, out3.shape[2], out3.shape[3])
            _c1 = self.cca1(_c1)
            _c2 = self.cca2(_c2)
            _c3 = self.cca3(_c3)
            _c4 = self.cca4(_c4)

            _c = self.linear_fuse(torch.cat([_c1,_c2,_c3,_c4],dim=1))

        elif self.attention == 'backbone_multi-levelv7-iii':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2,lateral_c4],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1,out1,lateral_c4],dim=1))
            _c = self.linear_fuse(torch.cat([lateral_c4,out1,out2,out3],dim=1))
        elif self.attention == 'backbone_multi-levelv7-iv':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2,lateral_c4],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1,out1],dim=1))
            _c = self.linear_fuse(torch.cat([lateral_c4,out1,out2,out3],dim=1))

        elif self.attention == 'backbone_multi-levelv7-1':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            # 做相减
            high = torch.cat([c1,lateral_c2,lateral_c3],dim=1)
            sub = high - lateral_c4
            sub = self.ffn_sub(sub)
            _c = self.linear_fuse(torch.cat([lateral_c4, out1, out2, out3,sub], dim=1))

        elif self.attention == 'backbone_multi-levelv7-2':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            # 做相减
            high = torch.cat([c1,lateral_c2,lateral_c3],dim=1)
            sub = high - lateral_c4
            cca_sub = self.cca_sub(sub)
            _c = self.linear_fuse(torch.cat([lateral_c4, out1, out2, out3,cca_sub], dim=1))
        elif self.attention == 'backbone_multi-levelv7-3':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4, lateral_c3], dim=1))
            #相减
            sub1 = lateral_c2 - out1
            out2 = self.ffn2(torch.cat([out1,sub1],dim=1))
            # 相减
            sub2 = c1 - out2
            out3 = self.ffn3(torch.cat([sub2,out2],dim=1))
            _c = self.linear_fuse(torch.cat([lateral_c4, out1, out2, out3], dim=1))

        elif self.attention == 'backbone_multi-levelv8':
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4,lateral_c3],dim=1))
            out2 = self.ffn2(torch.cat([out1,lateral_c2],dim=1))
            out3 = self.ffn3(torch.cat([out2,c1],dim=1))
            _c1 = self.cca1(out1)
            _c2 = self.cca2(out2)
            _c3 = self.cca3(out3)
            _c4 = self.cca4(lateral_c4)

            _c = self.linear_fuse(torch.cat([_c1,_c2,_c3,_c4],dim=1))


        elif self.attention == 'backbone_subv1':
            # 先统一通道
            lateral_c1 = self.lateral_c1(c1)
            lateral_c2 = self.lateral_c2(c2)
            lateral_c3 = self.lateral_c3(c3)
            lateral_c4 = self.lateral_c4(c4)
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(lateral_c2, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(lateral_c3, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(lateral_c4, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            # 做相减
            lateral_sub1 = lateral_c1 - lateral_c2
            lateral_sub2 = lateral_c3 - lateral_c4
            mlp_sub1 = _c1 - _c2
            mlp_sub2 = _c3 - _c4
            # 做crisscross attention
            cca_result1 = self.cca1.cross_forward(lateral_sub1,mlp_sub1)
            cca_result2 = self.cca1.cross_forward(lateral_sub2,mlp_sub2)
            result = self.linear_sub_fuse(torch.cat([cca_result1,cca_result2],dim=1))
            result = result.permute(0,2,1).reshape(n, -1, _c1.shape[2], _c1.shape[3])
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1,result], dim=1))
        elif self.attention == 'backbone_subv2':
            # 先统一通道
            lateral_c1 = self.lateral_c1(c1)
            lateral_c2 = self.lateral_c2(c2)
            lateral_c3 = self.lateral_c3(c3)
            lateral_c4 = self.lateral_c4(c4)
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(lateral_c2, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(lateral_c3, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(lateral_c4, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            # 做相减
            lateral_sub1 = lateral_c1 - lateral_c2
            lateral_sub2 = lateral_c3 - lateral_c4
            mlp_sub1 = _c1 - _c2
            mlp_sub2 = _c3 - _c4
            # 做crisscross attention
            cca_result1 = self.cca1.cross_forward(lateral_sub1, mlp_sub1)
            cca_result2 = self.cca1.cross_forward(lateral_sub2, mlp_sub2)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, cca_result1,cca_result2], dim=1))
        elif self.attention == 'backbone_addv1':
            # 先统一通道
            lateral_c1 = self.lateral_c1(c1)
            lateral_c2 = self.lateral_c2(c2)
            lateral_c3 = self.lateral_c3(c3)
            lateral_c4 = self.lateral_c4(c4)
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(lateral_c2, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(lateral_c3, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(lateral_c4, size=lateral_c1.size()[2:], mode='bilinear', align_corners=False)
            # 做相加
            lateral_sub1 = lateral_c1 + lateral_c2
            lateral_sub2 = lateral_c3 + lateral_c4
            mlp_sub1 = _c1 + _c2
            mlp_sub2 = _c3 + _c4

            # 做crisscross attention
            cca_result1 = self.cca1.cross_forward(lateral_sub1, mlp_sub1)
            cca_result2 = self.cca1.cross_forward(lateral_sub2, mlp_sub2)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, cca_result1,cca_result2], dim=1))
        else:
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        out_feat = self.dropout(_c)
        # x = self.classifier(out_feat)

        # return out_feat,x
        # return out_feat
        return {"out_feat":out_feat,
                "_c1":_c1,
                "_c2":_c2,
                "_c3":_c3,
                "_c4":_c4,}

class org_SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(org_SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

        self.classifier = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        backbone_feats = self.backbone.forward(inputs)
        # out_feat,out_classifier = self.decode_head.forward(backbone_feats)
        out_feat = self.decode_head.forward(backbone_feats)['out_feat']
        out_classifier = self.classifier(out_feat)

        x = F.interpolate(out_classifier, size=(H, W), mode='bilinear', align_corners=True)

        return {'out':x,
                'out_features':out_feat,
                'out_classifier':out_classifier,
                'backbone_features':backbone_feats}

class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False,seghead_last=False,attention=None):
        super(SegFormer, self).__init__()
        self.seghead_last = seghead_last
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim,attention=attention)

        # self.reduct4loss = ConvModule(
        #     c1=self.embedding_dim,
        #     c2=256,
        #     k=1,
        # )

        self.classifier = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1)
        # self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)



    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        backbone_feats = self.backbone.forward(inputs)
        # out_feat,out_classifier = self.decode_head.forward(backbone_feats)
        decodehead_out = self.decode_head.forward(backbone_feats)
        out_feat = decodehead_out['out_feat']
        # out_feat = self.reduct4loss(out_feat)
        if self.seghead_last:
            out_classifier = F.interpolate(out_feat, size=(H, W), mode='bilinear', align_corners=True)
            x = self.classifier(out_classifier)
        else:
            out_classifier = self.classifier(out_feat)

            x = F.interpolate(out_classifier, size=(H, W), mode='bilinear', align_corners=True)


        return {'out':x,
                'out_features':out_feat,
                'out_classifier':out_classifier,
                'decodehead_out':decodehead_out,
                'backbone_features':backbone_feats,
                'c3': backbone_feats[2],
                }


class SegFormerHead4Dualbackbone(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self,former_in_channels=[32, 64, 160, 256],resnet_in_channels = [64, 128, 256, 512], embedding_dim=768, dropout_ratio=0.1,
                 version='v1'):
        super(SegFormerHead4Dualbackbone, self).__init__()
        former_c1_in_channels, former_c2_in_channels, former_c3_in_channels, former_c4_in_channels = former_in_channels
        resnet_c1_in_channels, resnet_c2_in_channels, resnet_c3_in_channels, resnet_c4_in_channels = resnet_in_channels

        self.version = version
        if self.version == 'v1':
            self.former_linear_c4 = MLP(input_dim=former_c4_in_channels, embed_dim=embedding_dim)
            self.former_linear_c3 = MLP(input_dim=former_c3_in_channels, embed_dim=embedding_dim)
            self.former_linear_c2 = MLP(input_dim=former_c2_in_channels, embed_dim=embedding_dim)
            self.former_linear_c1 = MLP(input_dim=former_c1_in_channels, embed_dim=embedding_dim)
            #
            self.resnet_linear_c4 = MLP(input_dim=resnet_c4_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c3 = MLP(input_dim=resnet_c3_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c2 = MLP(input_dim=resnet_c2_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c1 = MLP(input_dim=resnet_c1_in_channels, embed_dim=embedding_dim)
            #
            self.resnet_linear_fuse = ConvModule(
                c1=embedding_dim * 4,
                c2=embedding_dim,
                k=1,
            )
            #
            self.former_linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif self.version == 'v1-1':
            self.former_linear_c4 = MLP(input_dim=former_c4_in_channels, embed_dim=embedding_dim)
            self.former_linear_c3 = MLP(input_dim=former_c3_in_channels, embed_dim=embedding_dim)
            self.former_linear_c2 = MLP(input_dim=former_c2_in_channels, embed_dim=embedding_dim)
            self.former_linear_c1 = MLP(input_dim=former_c1_in_channels, embed_dim=embedding_dim)
            #
            self.resnet_linear_c4 = MLP(input_dim=resnet_c4_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c3 = MLP(input_dim=resnet_c3_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c2 = MLP(input_dim=resnet_c2_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c1 = MLP(input_dim=resnet_c1_in_channels, embed_dim=embedding_dim)
            #
            self.resnet_linear_fuse = ConvModule(
                c1=embedding_dim * 4,
                c2=embedding_dim,
                k=1,
            )
            self.cca = CrissCrossAttention(embedding_dim)
            #
            self.former_linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        elif self.version == 'v2':
            self.former_linear_c4 = MLP(input_dim=former_c4_in_channels, embed_dim=embedding_dim)
            self.former_linear_c3 = MLP(input_dim=former_c3_in_channels, embed_dim=embedding_dim)
            self.former_linear_c2 = MLP(input_dim=former_c2_in_channels, embed_dim=embedding_dim)
            self.former_linear_c1 = MLP(input_dim=former_c1_in_channels, embed_dim=embedding_dim)
            #
            self.resnet_linear_c4 = MLP(input_dim=resnet_c4_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c3 = MLP(input_dim=resnet_c3_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c2 = MLP(input_dim=resnet_c2_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c1 = MLP(input_dim=resnet_c1_in_channels, embed_dim=embedding_dim)
            #
            self.cca_c1 = CrissCrossAttention(embedding_dim)
            self.cca_c2 = CrissCrossAttention(embedding_dim)
            self.cca_c3 = CrissCrossAttention(embedding_dim)
            self.cca_c4 = CrissCrossAttention(embedding_dim)
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 4,
                c2=embedding_dim,
                k=1,
            )
        elif self.version == 'v3':
            self.former_linear_c4 = MLP(input_dim=former_c4_in_channels, embed_dim=embedding_dim)
            self.former_linear_c3 = MLP(input_dim=former_c3_in_channels, embed_dim=embedding_dim)
            self.former_linear_c2 = MLP(input_dim=former_c2_in_channels, embed_dim=embedding_dim)
            self.former_linear_c1 = MLP(input_dim=former_c1_in_channels, embed_dim=embedding_dim)
            #
            self.resnet_linear_c4 = MLP(input_dim=resnet_c4_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c3 = MLP(input_dim=resnet_c3_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c2 = MLP(input_dim=resnet_c2_in_channels, embed_dim=embedding_dim)
            self.resnet_linear_c1 = MLP(input_dim=resnet_c1_in_channels, embed_dim=embedding_dim)
            #
            self.cca_c1 = CrissCrossAttention(embedding_dim)
            self.cca_c2 = CrissCrossAttention(embedding_dim)
            self.cca_c3 = CrissCrossAttention(embedding_dim)
            self.cca_c4 = CrissCrossAttention(embedding_dim)

            self.resnet_linear_fuse = ConvModule(
                c1=embedding_dim * 4,
                c2=embedding_dim,
                k=1,
            )
            self.linear_fuse = ConvModule(
                c1=embedding_dim * 5,
                c2=embedding_dim,
                k=1,
            )
        else:
            self.linear_c4 = MLP(input_dim=former_c4_in_channels+resnet_c4_in_channels, embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=former_c3_in_channels+resnet_c3_in_channels, embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=former_c2_in_channels+resnet_c2_in_channels, embed_dim=embedding_dim)
            self.linear_c1 = MLP(input_dim=former_c1_in_channels+resnet_c1_in_channels, embed_dim=embedding_dim)



            self.linear_fuse = ConvModule(
                c1=embedding_dim * 4,
                c2=embedding_dim,
                k=1,
            )

        # self.classifier    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, former_backbone_feats,resnet_backbone_feats):
        former_c1, former_c2, former_c3, former_c4 = former_backbone_feats
        resnet_c1, resnet_c2, resnet_c3, resnet_c4 = resnet_backbone_feats['c1'],resnet_backbone_feats['c2'],resnet_backbone_feats['c3'],resnet_backbone_feats['c4']
        if self.version == 'v1':
            n, _, h, w = former_c4.shape

            _former_c4 = self.former_linear_c4(former_c4).permute(0, 2, 1).reshape(n, -1, former_c4.shape[2], former_c4.shape[3])
            _former_c4 = F.interpolate(_former_c4, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c3 = self.former_linear_c3(former_c3).permute(0, 2, 1).reshape(n, -1, former_c3.shape[2], former_c3.shape[3])
            _former_c3 = F.interpolate(_former_c3, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c2 = self.former_linear_c2(former_c2).permute(0, 2, 1).reshape(n, -1, former_c2.shape[2], former_c2.shape[3])
            _former_c2 = F.interpolate(_former_c2, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c1 = self.former_linear_c1(former_c1).permute(0, 2, 1).reshape(n, -1, former_c1.shape[2], former_c1.shape[3])

            #

            _resnet_c4 = self.resnet_linear_c4(resnet_c4).permute(0, 2, 1).reshape(n, -1, resnet_c4.shape[2], resnet_c4.shape[3])
            _resnet_c4 = F.interpolate(_resnet_c4, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c3 = self.resnet_linear_c3(resnet_c3).permute(0, 2, 1).reshape(n, -1, resnet_c3.shape[2], resnet_c3.shape[3])
            _resnet_c3 = F.interpolate(_resnet_c3, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c2 = self.resnet_linear_c2(resnet_c2).permute(0, 2, 1).reshape(n, -1, resnet_c2.shape[2], resnet_c2.shape[3])
            _resnet_c2 = F.interpolate(_resnet_c2, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c1 = self.resnet_linear_c1(resnet_c1).permute(0, 2, 1).reshape(n, -1, resnet_c1.shape[2], resnet_c1.shape[3])

            _resnet_c = self.resnet_linear_fuse(torch.cat([_resnet_c4, _resnet_c3, _resnet_c2, _resnet_c1], dim=1))
            _c = self.former_linear_fuse(torch.cat([_resnet_c4, _resnet_c3, _resnet_c2, _resnet_c1,_resnet_c], dim=1))
            _c1,_c2,_c3,_c4 = _former_c1,_former_c2,_former_c3,_former_c4
        elif self.version == 'v1-1':
            n, _, h, w = former_c4.shape

            _former_c4 = self.former_linear_c4(former_c4).permute(0, 2, 1).reshape(n, -1, former_c4.shape[2], former_c4.shape[3])
            _former_c4 = F.interpolate(_former_c4, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c3 = self.former_linear_c3(former_c3).permute(0, 2, 1).reshape(n, -1, former_c3.shape[2], former_c3.shape[3])
            _former_c3 = F.interpolate(_former_c3, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c2 = self.former_linear_c2(former_c2).permute(0, 2, 1).reshape(n, -1, former_c2.shape[2], former_c2.shape[3])
            _former_c2 = F.interpolate(_former_c2, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c1 = self.former_linear_c1(former_c1).permute(0, 2, 1).reshape(n, -1, former_c1.shape[2], former_c1.shape[3])

            #

            _resnet_c4 = self.resnet_linear_c4(resnet_c4).permute(0, 2, 1).reshape(n, -1, resnet_c4.shape[2], resnet_c4.shape[3])
            _resnet_c4 = F.interpolate(_resnet_c4, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c3 = self.resnet_linear_c3(resnet_c3).permute(0, 2, 1).reshape(n, -1, resnet_c3.shape[2], resnet_c3.shape[3])
            _resnet_c3 = F.interpolate(_resnet_c3, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c2 = self.resnet_linear_c2(resnet_c2).permute(0, 2, 1).reshape(n, -1, resnet_c2.shape[2], resnet_c2.shape[3])
            _resnet_c2 = F.interpolate(_resnet_c2, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c1 = self.resnet_linear_c1(resnet_c1).permute(0, 2, 1).reshape(n, -1, resnet_c1.shape[2], resnet_c1.shape[3])

            _resnet_c = self.resnet_linear_fuse(torch.cat([_resnet_c4, _resnet_c3, _resnet_c2, _resnet_c1], dim=1))
            _resnet_c = self.cca(_resnet_c)
            _c = self.former_linear_fuse(torch.cat([_resnet_c4, _resnet_c3, _resnet_c2, _resnet_c1,_resnet_c], dim=1))
            _c1,_c2,_c3,_c4 = _former_c1,_former_c2,_former_c3,_former_c4

        elif self.version == 'v2':
            n, _, h, w = former_c4.shape

            _former_c4 = self.former_linear_c4(former_c4).permute(0, 2, 1).reshape(n, -1, former_c4.shape[2],
                                                                                   former_c4.shape[3])
            _former_c4 = F.interpolate(_former_c4, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c3 = self.former_linear_c3(former_c3).permute(0, 2, 1).reshape(n, -1, former_c3.shape[2],
                                                                                   former_c3.shape[3])
            _former_c3 = F.interpolate(_former_c3, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c2 = self.former_linear_c2(former_c2).permute(0, 2, 1).reshape(n, -1, former_c2.shape[2],
                                                                                   former_c2.shape[3])
            _former_c2 = F.interpolate(_former_c2, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c1 = self.former_linear_c1(former_c1).permute(0, 2, 1).reshape(n, -1, former_c1.shape[2],
                                                                                   former_c1.shape[3])

            #

            _resnet_c4 = self.resnet_linear_c4(resnet_c4).permute(0, 2, 1).reshape(n, -1, resnet_c4.shape[2],
                                                                                   resnet_c4.shape[3])
            _resnet_c4 = F.interpolate(_resnet_c4, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c3 = self.resnet_linear_c3(resnet_c3).permute(0, 2, 1).reshape(n, -1, resnet_c3.shape[2],
                                                                                   resnet_c3.shape[3])
            _resnet_c3 = F.interpolate(_resnet_c3, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c2 = self.resnet_linear_c2(resnet_c2).permute(0, 2, 1).reshape(n, -1, resnet_c2.shape[2],
                                                                                   resnet_c2.shape[3])
            _resnet_c2 = F.interpolate(_resnet_c2, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c1 = self.resnet_linear_c1(resnet_c1).permute(0, 2, 1).reshape(n, -1, resnet_c1.shape[2],
                                                                                   resnet_c1.shape[3])
            _cca1 = self.cca_c1.cross_forward(_resnet_c1,_former_c1)
            _cca2 = self.cca_c1.cross_forward(_resnet_c2,_former_c2)
            _cca3 = self.cca_c1.cross_forward(_resnet_c3,_former_c3)
            _cca4 = self.cca_c1.cross_forward(_resnet_c4,_former_c4)
            _c = self.linear_fuse(torch.cat([_cca1,_cca2,_cca3,_cca4], dim=1))
            _c1, _c2, _c3, _c4 = _cca1, _cca2, _cca3, _cca4
        elif self.version == 'v3':
            n, _, h, w = former_c4.shape

            _former_c4 = self.former_linear_c4(former_c4).permute(0, 2, 1).reshape(n, -1, former_c4.shape[2],
                                                                                   former_c4.shape[3])
            _former_c4 = F.interpolate(_former_c4, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c3 = self.former_linear_c3(former_c3).permute(0, 2, 1).reshape(n, -1, former_c3.shape[2],
                                                                                   former_c3.shape[3])
            _former_c3 = F.interpolate(_former_c3, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c2 = self.former_linear_c2(former_c2).permute(0, 2, 1).reshape(n, -1, former_c2.shape[2],
                                                                                   former_c2.shape[3])
            _former_c2 = F.interpolate(_former_c2, size=former_c1.size()[2:], mode='bilinear', align_corners=False)

            _former_c1 = self.former_linear_c1(former_c1).permute(0, 2, 1).reshape(n, -1, former_c1.shape[2],
                                                                                   former_c1.shape[3])

            #

            _resnet_c4 = self.resnet_linear_c4(resnet_c4).permute(0, 2, 1).reshape(n, -1, resnet_c4.shape[2],
                                                                                   resnet_c4.shape[3])
            _resnet_c4 = F.interpolate(_resnet_c4, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c3 = self.resnet_linear_c3(resnet_c3).permute(0, 2, 1).reshape(n, -1, resnet_c3.shape[2],
                                                                                   resnet_c3.shape[3])
            _resnet_c3 = F.interpolate(_resnet_c3, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c2 = self.resnet_linear_c2(resnet_c2).permute(0, 2, 1).reshape(n, -1, resnet_c2.shape[2],
                                                                                   resnet_c2.shape[3])
            _resnet_c2 = F.interpolate(_resnet_c2, size=resnet_c1.size()[2:], mode='bilinear', align_corners=False)

            _resnet_c1 = self.resnet_linear_c1(resnet_c1).permute(0, 2, 1).reshape(n, -1, resnet_c1.shape[2],
                                                                                   resnet_c1.shape[3])
            _cca1 = self.cca_c1.cross_forward(_resnet_c1,_former_c1)
            _cca2 = self.cca_c1.cross_forward(_resnet_c2,_former_c2)
            _cca3 = self.cca_c1.cross_forward(_resnet_c3,_former_c3)
            _cca4 = self.cca_c1.cross_forward(_resnet_c4,_former_c4)
            _resnet_fuse_c = self.resnet_linear_fuse(torch.cat([_cca1,_cca2,_cca3,_cca4], dim=1))
            _c = self.linear_fuse(torch.cat([_former_c1,_former_c2,_former_c3,_former_c4,_resnet_fuse_c], dim=1))
            _c1, _c2, _c3, _c4 = _cca1, _cca2, _cca3, _cca4
        else:
            c1 = torch.cat([former_c1,resnet_c1],dim=1)
            c2 = torch.cat([former_c2,resnet_c2],dim=1)
            c3 = torch.cat([former_c3,resnet_c3],dim=1)
            c4 = torch.cat([former_c4,resnet_c4],dim=1)

            ############## MLP decoder on C1-C4 ###########
            n, _, h, w = c4.shape

            _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
            _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

            _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

            _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
            _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

            _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

            #
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        out_feat = self.dropout(_c)
        # x = self.classifier(out_feat)

        # return out_feat,x
        # return out_feat
        return {"out_feat": out_feat,
                "_c1": _c1,
                "_c2": _c2,
                "_c3": _c3,
                "_c4": _c4, }


class dual_backbones(nn.Module):
    def __init__(self,phi = 'b0',res='resnet34',pretrained = False):
        super(dual_backbones, self).__init__()
        self.backbone_former = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,}[phi](pretrained)
        self.backbone_resnet = {
            'resnet18':resnet18,'resnet34':resnet34
        }[res](pretrained=pretrained)
    def forward(self):
        pass

class ResSegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0',res='resnet34', pretrained = False,seghead_last=False,version=None):
        super(ResSegFormer, self).__init__()
        self.seghead_last = seghead_last
        self.former_in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.resnet_in_channels = {
            'resnet18': [64, 128, 256, 512], 'resnet34': [64, 128, 256, 512]
        }[res]
        self.backbone   = dual_backbones(phi=phi,res=res,pretrained=pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead4Dualbackbone(self.former_in_channels,self.resnet_in_channels, self.embedding_dim,version=version)

        # self.reduct4loss = ConvModule(
        #     c1=self.embedding_dim,
        #     c2=256,
        #     k=1,
        # )

        self.classifier = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1)
        # self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)



    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        former_backbone_feats = self.backbone.backbone_former.forward(inputs)
        resnet_backbone_feats = self.backbone.backbone_resnet.base_forward(inputs)
        decodehead_out = self.decode_head.forward(former_backbone_feats,resnet_backbone_feats)
        out_feat = decodehead_out['out_feat']
        # out_feat = self.reduct4loss(out_feat)
        if self.seghead_last:
            out_classifier = F.interpolate(out_feat, size=(H, W), mode='bilinear', align_corners=True)
            x = self.classifier(out_classifier)
        else:
            out_classifier = self.classifier(out_feat)

            x = F.interpolate(out_classifier, size=(H, W), mode='bilinear', align_corners=True)


        return {'out':x,
                'out_features':out_feat,
                'out_classifier':out_classifier,
                'decodehead_out':decodehead_out,
                'backbone_features':former_backbone_feats,
                'c3': former_backbone_feats[2],
                }


if __name__ == '__main__':
    # ckpt_path = '../../../pretrained/segformer_b2_weights_voc.pth'
    # sd = torch.load(ckpt_path,map_location='cpu')

    # model = ResSegFormer(num_classes=3, phi='b2',res='resnet34', pretrained=False,version='v2')
    model = SegFormer(num_classes=3, phi='b2', pretrained=False,attention='backbone_multi-levelv7-ii-1-7')
    img = torch.randn(2,3,256,256)
    out = model(img)
    logits = out['out']
    # print(logits.shape)