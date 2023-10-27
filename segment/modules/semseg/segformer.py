# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment.modules.semseg.nn import Attention,CrissCrossAttention,CoordAtt,CBAMBlock
from segment.modules.backbone.resnet import resnet18,resnet34, resnet50, resnet101
from segment.modules.backbone.mit import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from segment.demo.DAM.dam import DAM,DAM_criss,AxialDAM
from segment.demo.gold_yolo.Low_FAMIFM import FAMIFM
from segment.demo.gold_yolo.transformer import InjectionMultiSum_Auto_pool,Skip_InjectionMultiSum_Auto_pool

from segment.segment_anything_main.segment_anything import sam_model_registry,SamPredictor





class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

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




        if attention == 'backbone_multi-levelv7-ii-1-6-v1':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)


            self.linear_c1 = MLP(input_dim=c1_in_channels,embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels,embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels,embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=c4_in_channels,embed_dim=embedding_dim)

            self.ffn_sub = ConvModule(embedding_dim * 2,embedding_dim)
            self.cca1 = nn.Sequential(
                CrissCrossAttention(embedding_dim),
                CrissCrossAttention(embedding_dim)
            )

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-6-v1-dam':
            self.ffn1 = ConvModule(c3_in_channels+c4_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c3_in_channels+c2_in_channels,c2_in_channels)
            self.ffn3 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)

            self.linear_c1 = MLP(input_dim=c1_in_channels,embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels,embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels,embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=c4_in_channels,embed_dim=embedding_dim)

            self.dam = DAM(c4_in_channels)
            self.ffn_sub = ConvModule(embedding_dim * 2,embedding_dim)
            self.cca1 = nn.Sequential(
                CrissCrossAttention(embedding_dim),
                CrissCrossAttention(embedding_dim)
            )

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'backbone_multi-levelv7-ii-1-6-v1-dam-criss':
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels + c4_in_channels),
                ConvModule(c3_in_channels + c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c3_in_channels+c2_in_channels),
                ConvModule(c3_in_channels+c2_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c2_in_channels+c1_in_channels),
                ConvModule(c2_in_channels+c1_in_channels, c1_in_channels,k=3,p=1)
            )

            self.linear_c1 = MLP(input_dim=c1_in_channels,embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels,embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels,embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=c4_in_channels,embed_dim=embedding_dim)

            self.dam = DAM(c4_in_channels)
            self.ffn_sub = ConvModule(embedding_dim * 2,embedding_dim)
            self.cca1 = nn.Sequential(
                CrissCrossAttention(embedding_dim),
                CrissCrossAttention(embedding_dim)
            )

            self.linear_fuse = ConvModule(
                c1=embedding_dim*5,
                c2=embedding_dim,
                k=1,
            )
        elif attention == 'o1':
            self.dam = DAM(c4_in_channels)
            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-cbam':
            self.f4_dam = DAM(c4_in_channels)
            self.f3_dam = CBAMBlock(c3_in_channels)
            self.f2_dam = CBAMBlock(c2_in_channels)
            self.f1_dam = CBAMBlock(c1_in_channels)

            self.ffn3 = ConvModule(c4_in_channels+c3_in_channels,c3_in_channels)
            self.ffn2 = ConvModule(c2_in_channels+c3_in_channels,c2_in_channels)
            self.ffn1 = ConvModule(c2_in_channels+c1_in_channels,c1_in_channels)

            self.ffn = nn.Sequential(
                ConvModule(c4_in_channels+c3_in_channels + c2_in_channels + c1_in_channels, 64)
            )



        elif attention == 'o1-damcriss':
            self.dam = DAM_criss(c4_in_channels)
            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam':
            self.dam = DAM(c4_in_channels)
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels,c2_in_channels,c3_in_channels,c4_in_channels])
            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam-inj':
            self.dam = DAM(c4_in_channels)
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels,c2_in_channels,c3_in_channels,c4_in_channels])
            self.inj3 = InjectionMultiSum_Auto_pool(c3_in_channels,c3_in_channels,activations=nn.ReLU6)
            self.inj2 = InjectionMultiSum_Auto_pool(c2_in_channels,c2_in_channels,activations=nn.ReLU6)
            self.inj1 = InjectionMultiSum_Auto_pool(c1_in_channels,c1_in_channels,activations=nn.ReLU6)
            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam-inj-v1':
            self.dam = DAM(c4_in_channels)
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])

            self.inj3 = InjectionMultiSum_Auto_pool(c3_in_channels, c3_in_channels, activations=nn.ReLU6)
            self.inj2 = InjectionMultiSum_Auto_pool(c2_in_channels, c2_in_channels, activations=nn.ReLU6)
            self.inj1 = InjectionMultiSum_Auto_pool(c1_in_channels, c1_in_channels, activations=nn.ReLU6)

            self.fuse_3_4 = nn.Sequential(
                ConvModule(c3_in_channels + c4_in_channels, c3_in_channels, k=3, p=1)
            )

            self.ffn = nn.Sequential(
                ConvModule(c3_in_channels + c2_in_channels + c1_in_channels, 64, k=3, p=1)
            )
        elif attention == 'o1-fam-inj-v1-cbam':
            self.dam = DAM(c4_in_channels)
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])

            self.inj3 = InjectionMultiSum_Auto_pool(c3_in_channels, c3_in_channels, activations=nn.ReLU6)
            self.inj2 = InjectionMultiSum_Auto_pool(c2_in_channels, c2_in_channels, activations=nn.ReLU6)
            self.inj1 = InjectionMultiSum_Auto_pool(c1_in_channels, c1_in_channels, activations=nn.ReLU6)

            self.f1_cbam = CBAMBlock(c1_in_channels)
            self.f2_cbam = CBAMBlock(c2_in_channels)
            self.f3_cbam = CBAMBlock(c3_in_channels)
            self.fuse_3_4 = nn.Sequential(
                ConvModule(c3_in_channels + c4_in_channels, c3_in_channels, k=3, p=1)
            )

            self.ffn = nn.Sequential(
                ConvModule(c3_in_channels + c2_in_channels + c1_in_channels, 64, k=3, p=1)
            )
        elif attention == 'o1-fam-inj-skip-all-no':
            self.ffn0 = nn.Sequential(
                ConvModule(c4_in_channels, c3_in_channels)
            )
            self.ffn1 = nn.Sequential(
                ConvModule(c3_in_channels, c2_in_channels)
            )
            self.ffn2 = nn.Sequential(
                ConvModule(c2_in_channels, c1_in_channels)
            )
            self.ffn3 = nn.Sequential(
                ConvModule(c1_in_channels, 64)
            )
        elif attention == 'dec_transpose':
            self.center = _DecoderBlock(c4_in_channels,1024,c4_in_channels)
            self.dec4 = _DecoderBlock(c4_in_channels + c4_in_channels,c4_in_channels,256)
            self.dec3 = _DecoderBlock(256 + c3_in_channels,384,c2_in_channels)
            self.dec2 = _DecoderBlock(c2_in_channels + c2_in_channels,c2_in_channels,c1_in_channels)
            self.dec1 = _DecoderBlock(c1_in_channels * 2,c1_in_channels * 2,64)
        elif attention == 'dec_transpose_decCBAM':
            self.center = _DecoderBlock(c4_in_channels,1024,c4_in_channels)
            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels,c4_in_channels,256),
                                      CBAMBlock(256)
                                      )
            self.dec3 = nn.Sequential(_DecoderBlock(256 + c3_in_channels,384,c2_in_channels),
                                      CBAMBlock(c2_in_channels)
                                      )
            self.dec2 = nn.Sequential(_DecoderBlock(c2_in_channels + c2_in_channels,c2_in_channels,c1_in_channels),
                                      CBAMBlock(c1_in_channels)
                                      )
            self.dec1 = nn.Sequential(_DecoderBlock(c1_in_channels * 2,c1_in_channels * 2,64),
                                      CBAMBlock(64)
                                      )
        elif attention == 'dec_transpose_FAMIFM':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels,c2_in_channels,c3_in_channels,c4_in_channels])
            self.center = _DecoderBlock(c4_in_channels,1024,c4_in_channels)

            self.dec4 = _DecoderBlock(c4_in_channels + c4_in_channels,c4_in_channels,256)
            self.dec3 = _DecoderBlock(256 + c3_in_channels + c3_in_channels,384,c2_in_channels)
            self.dec2 = _DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels,c2_in_channels,c1_in_channels)
            self.dec1 = _DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels,c1_in_channels * 2,64)
        elif attention == 'dec_transpose_FAMIFM_CBAM':
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels,c2_in_channels,c3_in_channels,c4_in_channels])
            self.center = _DecoderBlock(c4_in_channels,1024,c4_in_channels)

            self.dec4 = nn.Sequential(
                _DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256),
                CBAMBlock(256)
            )
            self.dec3 = nn.Sequential(
                _DecoderBlock(256 + c3_in_channels + c3_in_channels,384,c2_in_channels),
                CBAMBlock(c2_in_channels)
            )
            self.dec2 = nn.Sequential(
                _DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels,c2_in_channels,c1_in_channels),
                CBAMBlock(c1_in_channels)
            )
            self.dec1 = nn.Sequential(
                _DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels,c1_in_channels * 2,64),
                CBAMBlock(64)
            )
        elif attention == 'dec_transpose_FAMIFM_decCCA':
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = _DecoderBlock(c4_in_channels, 1024, c4_in_channels)

            self.dec4 = nn.Sequential(
                _DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256),
                CrissCrossAttention(256),
                CrissCrossAttention(256),
            )
            self.dec3 = nn.Sequential(
                _DecoderBlock(256 + c3_in_channels + c3_in_channels, 384, c2_in_channels),
                CrissCrossAttention(c2_in_channels),
                CrissCrossAttention(c2_in_channels),

            )
            self.dec2 = nn.Sequential(
                _DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels),
                CrissCrossAttention(c1_in_channels),
                CrissCrossAttention(c1_in_channels),
            )
            self.dec1 = nn.Sequential(
                _DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels, c1_in_channels * 2, 64),
                CrissCrossAttention(64),
                CrissCrossAttention(64),
            )
        elif attention == 'dec_transpose_DAM':
            # self.center = _DecoderBlock(c4_in_channels,1024,c4_in_channels)
            self.center = DAM(c4_in_channels)
            self.dec4 = _DecoderBlock(c4_in_channels + c4_in_channels,c4_in_channels,256)
            self.dec3 = _DecoderBlock(256 + c3_in_channels,384,c2_in_channels)
            self.dec2 = _DecoderBlock(c2_in_channels + c2_in_channels,c2_in_channels,c1_in_channels)
            self.dec1 = _DecoderBlock(c1_in_channels * 2,c1_in_channels * 2,64)
        elif attention == 'dec_transpose_AxialDAM':
            # self.center = _DecoderBlock(c4_in_channels,1024,c4_in_channels)
            self.center = AxialDAM(c4_in_channels)
            self.dec4 = _DecoderBlock(c4_in_channels + c4_in_channels,c4_in_channels,256)
            self.dec3 = _DecoderBlock(256 + c3_in_channels,384,c2_in_channels)
            self.dec2 = _DecoderBlock(c2_in_channels + c2_in_channels,c2_in_channels,c1_in_channels)
            self.dec1 = _DecoderBlock(c1_in_channels * 2,c1_in_channels * 2,64)
        elif attention == 'dec_transpose_CCA_dec_CCA':
            self.center = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                CrissCrossAttention(c4_in_channels)
            )
            self.dec4 = nn.Sequential(
                _DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256),
                CrissCrossAttention(256),
                CrissCrossAttention(256),
                )
            self.dec3 = nn.Sequential(
                _DecoderBlock(256 + c3_in_channels, 384, c2_in_channels),
                CrissCrossAttention(c2_in_channels),
                CrissCrossAttention(c2_in_channels),
                )
            self.dec2 = nn.Sequential(
                _DecoderBlock(c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels),
                CrissCrossAttention(c1_in_channels),
                CrissCrossAttention(c1_in_channels),
                )
            self.dec1 = nn.Sequential(
                _DecoderBlock(c1_in_channels * 2, c1_in_channels * 2, 64),
                CrissCrossAttention(64),
                CrissCrossAttention(64),
                )
        elif attention == 'dec_transpose_CCA':
            # self.center = DAM_criss(c4_in_channels)
            # self.dec4 = nn.Sequential(
            #     _DecoderBlock(c4_in_channels + c4_in_channels,c4_in_channels,256),
            #     CBAMBlock(256)
            # )
            # self.dec3 = nn.Sequential(
            #     _DecoderBlock(256 + c3_in_channels,384,c2_in_channels),
            #     CBAMBlock(c2_in_channels)
            # )
            # self.dec2 = nn.Sequential(
            #     _DecoderBlock(c2_in_channels + c2_in_channels,c2_in_channels,c1_in_channels),
            #     CBAMBlock(c1_in_channels)
            # )
            # self.dec1 = nn.Sequential(
            #     _DecoderBlock(c1_in_channels * 2,c1_in_channels * 2,64),
            #     CBAMBlock(64)
            # )
            self.center = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                CrissCrossAttention(c4_in_channels)
            )
            self.dec4 = _DecoderBlock(c4_in_channels + c4_in_channels,c4_in_channels,256)
            self.dec3 = _DecoderBlock(256 + c3_in_channels,384,c2_in_channels)
            self.dec2 = _DecoderBlock(c2_in_channels + c2_in_channels,c2_in_channels,c1_in_channels)
            self.dec1 = _DecoderBlock(c1_in_channels * 2,c1_in_channels * 2,64)
        elif attention == 'dec_transpose_FAMIFM_AxialDAM':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = AxialDAM(c4_in_channels)

            self.dec4 = _DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256)
            self.dec3 = _DecoderBlock(256 + c3_in_channels + c3_in_channels, 384, c2_in_channels)
            self.dec2 = _DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels)
            self.dec1 = _DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels, c1_in_channels * 2, 64)
        elif attention == 'dec_transpose_FAMIFM_DAM':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = DAM(c4_in_channels)

            self.dec4 = _DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256)
            self.dec3 = _DecoderBlock(256 + c3_in_channels + c3_in_channels, 384, c2_in_channels)
            self.dec2 = _DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels)
            self.dec1 = _DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels, c1_in_channels * 2, 64)
        elif attention == 'dec_transpose_FAMIFM_CBAM_DAM' or \
                attention == 'dec_transpose_FAMIFM_center_CBAM_DAM':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = DAM(c4_in_channels)

            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256),
                                      CBAMBlock(256)
                                      )
            self.dec3 = nn.Sequential(_DecoderBlock(256 + c3_in_channels + c3_in_channels, 384, c2_in_channels),
                                      CBAMBlock(c2_in_channels)
                                      )
            self.dec2 = nn.Sequential(_DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels),
                                      CBAMBlock(c1_in_channels)
                                      )
            self.dec1 = nn.Sequential(_DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels, c1_in_channels * 2, 64),
                                      CBAMBlock(64)
                                      )
        elif attention == 'dec_transpose_FAMIFM_CBAM_DAM_Inj':
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = DAM(c4_in_channels)

            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, c3_in_channels),
                                      CBAMBlock(c3_in_channels)
                                      )
            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels, c3_in_channels, activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels, c2_in_channels, activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels, c1_in_channels, activations=nn.ReLU6)

            self.dec3 = nn.Sequential(_DecoderBlock(c3_in_channels,c3_in_channels, c2_in_channels),
                                      CBAMBlock(c2_in_channels)
                                      )
            self.dec2 = nn.Sequential(
                _DecoderBlock(c2_in_channels, c2_in_channels, c1_in_channels),
                CBAMBlock(c1_in_channels)
                )
            self.dec1 = nn.Sequential(
                _DecoderBlock(c1_in_channels, c1_in_channels , 64),
                CBAMBlock(64)
                )


        elif attention == 'dec_transpose_FAMIFM_CBAM_CCA_Inj':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                CrissCrossAttention(c4_in_channels)
            )

            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, c3_in_channels),
                                      CBAMBlock(c3_in_channels)
                                      )
            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels, c3_in_channels, activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels, c2_in_channels, activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels, c1_in_channels, activations=nn.ReLU6)
            self.dec3 = nn.Sequential(_DecoderBlock(c3_in_channels, c3_in_channels, c2_in_channels),
                                      CBAMBlock(c2_in_channels)
                                      )
            self.dec2 = nn.Sequential(
                _DecoderBlock(c2_in_channels, c2_in_channels, c1_in_channels),
                CBAMBlock(c1_in_channels)
                )
            self.dec1 = nn.Sequential(
                _DecoderBlock(c1_in_channels , c1_in_channels , 64),
                CBAMBlock(64)
                )
        elif attention == 'dec_transpose_FAMIFM_CBAM_CCA_R2':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = CrissCrossAttention(c4_in_channels)

            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256),
                                      CBAMBlock(256)
                                      )

            self.dec3 = nn.Sequential(_DecoderBlock(256 + c3_in_channels + c3_in_channels, 384, c2_in_channels),
                                      CBAMBlock(c2_in_channels)
                                      )
            self.dec2 = nn.Sequential(
                _DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels),
                CBAMBlock(c1_in_channels)
                )
            self.dec1 = nn.Sequential(
                _DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels, c1_in_channels * 2, 64),
                CBAMBlock(64)
                )
        elif attention == 'dec_transpose_FAMIFM_CCA':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                CrissCrossAttention(c4_in_channels)
            )

            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256),
                                      )

            self.dec3 = nn.Sequential(_DecoderBlock(256 + c3_in_channels + c3_in_channels, 384, c2_in_channels),
                                      )
            self.dec2 = nn.Sequential(
                _DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels),
                )
            self.dec1 = nn.Sequential(
                _DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels, c1_in_channels * 2, 64),
                )
        elif attention == 'dec_transpose_FAMIFM_CBAM_CCA_SAM':
            # sam_checkpoint = "pretrained/sam_vit_h_4b8939.pth"
            # model_type = "vit_h"
            # self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            # self.predictor = SamPredictor(self.sam)
            # # Freeze the parameters of sam and predictor
            # for param in self.sam.parameters():
            #     param.requires_grad = False
            # for param in self.predictor.parameters():
            #     param.requires_grad = False


            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                CrissCrossAttention(c4_in_channels)
            )
            # self.center = DAM_criss(c4_in_channels)

            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256),
                                      CBAMBlock(256)
                                      )

            self.dec3 = nn.Sequential(_DecoderBlock(256 + c3_in_channels + c3_in_channels, 384, c2_in_channels),
                                      CBAMBlock(c2_in_channels)
                                      )
            self.dec2 = nn.Sequential(
                _DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels),
                CBAMBlock(c1_in_channels)
                )
            self.dec1 = nn.Sequential(
                _DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels, c1_in_channels * 2, 64),
                CBAMBlock(64)
                )
        elif attention == 'dec_transpose_FAMIFM_CBAM_CCA':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                CrissCrossAttention(c4_in_channels)
            )
            # self.center = DAM_criss(c4_in_channels)

            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, 256),
                                      CBAMBlock(256)
                                      )

            self.dec3 = nn.Sequential(_DecoderBlock(256 + c3_in_channels + c3_in_channels, 384, c2_in_channels),
                                      CBAMBlock(c2_in_channels)
                                      )
            self.dec2 = nn.Sequential(_DecoderBlock(c2_in_channels + c2_in_channels + c2_in_channels, c2_in_channels, c1_in_channels),
                                      CBAMBlock(c1_in_channels)
                                      )
            self.dec1 = nn.Sequential(_DecoderBlock(c1_in_channels + c1_in_channels + c1_in_channels, c1_in_channels * 2, 64),
                                      CBAMBlock(64)
                                      )
        elif attention == 'dec_transpose_FAMIFM_DAM_Inj':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = DAM(c4_in_channels)
            self.dec4 = _DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, c3_in_channels)

            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels, c3_in_channels, activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels, c2_in_channels, activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels, c1_in_channels, activations=nn.ReLU6)

            self.dec3 = _DecoderBlock(c3_in_channels, c3_in_channels, c2_in_channels)
            self.dec2 = _DecoderBlock(c2_in_channels, c2_in_channels, c1_in_channels)
            self.dec1 = _DecoderBlock(c1_in_channels, c1_in_channels, 64)
        elif attention == 'dec_transpose_FAMIFM_CBAM_Inj':
            # 在此条件下，FAMIFM推出来的特征，直接和_DecoderBlock之前的特征concatenate
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])
            self.center = _DecoderBlock(c4_in_channels,1024,c4_in_channels)
            self.dec4 = nn.Sequential(_DecoderBlock(c4_in_channels + c4_in_channels, c4_in_channels, c3_in_channels),
                                      CBAMBlock(c3_in_channels)
                                      )

            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels, c3_in_channels, activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels, c2_in_channels, activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels, c1_in_channels, activations=nn.ReLU6)

            self.dec3 = nn.Sequential(_DecoderBlock(c3_in_channels, c3_in_channels, c2_in_channels),
                                      CBAMBlock(c2_in_channels)
                                      )
            self.dec2 = nn.Sequential(_DecoderBlock(c2_in_channels, c2_in_channels, c1_in_channels),
                                      CBAMBlock(c1_in_channels)
                                      )
            self.dec1 = nn.Sequential(_DecoderBlock(c1_in_channels, c1_in_channels, 64),
                                      CBAMBlock(64)
                                      )

        elif attention == 'o1-fam-inj-skip':
            self.dam = DAM(c4_in_channels)
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels,c2_in_channels,c3_in_channels,c4_in_channels])

            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels,c3_in_channels,activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels,c2_in_channels,activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels,c1_in_channels,activations=nn.ReLU6)
            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam-inj-skip-no-dam':
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels])

            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels, c3_in_channels, activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels, c2_in_channels, activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels, c1_in_channels, activations=nn.ReLU6)
            self.ffn0 = nn.Sequential(
                ConvModule(c4_in_channels, c3_in_channels, k=3, p=1)
            )
            self.ffn1 = nn.Sequential(
                # CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels, k=3, p=1)
            )
            self.ffn2 = nn.Sequential(
                # CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels, k=3, p=1)
            )
            self.ffn3 = nn.Sequential(
                # CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64, k=3, p=1)
            )
        elif attention == 'o1-fam-inj-skip-no-FAM-IFM':
            self.dam = DAM(c4_in_channels)

            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels,c3_in_channels,activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels,c2_in_channels,activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels,c1_in_channels,activations=nn.ReLU6)
            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam-inj-skip-no-FAM-IFM-inj':
            self.dam = DAM(c4_in_channels)

            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam-inj-skip-no-dam-FAM-IFM':
            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels,c3_in_channels,activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels,c2_in_channels,activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels,c1_in_channels,activations=nn.ReLU6)
            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam-inj-skip-no-inj':
            self.dam = DAM(c4_in_channels)
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels,c2_in_channels,c3_in_channels,c4_in_channels])

            self.ffn0 = nn.Sequential(
                CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam-inj-skip-no-dam-inj':
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels,c2_in_channels,c3_in_channels,c4_in_channels])

            self.ffn0 = nn.Sequential(
                # CrissCrossAttention(c4_in_channels),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                # CrissCrossAttention(c3_in_channels),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                # CrissCrossAttention(c2_in_channels),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                # CrissCrossAttention(c1_in_channels),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        elif attention == 'o1-fam-inj-cbam-skip':
            self.dam = DAM(c4_in_channels)
            self.low_FAM_IFM = FAMIFM(fusion_in=c2_in_channels + c1_in_channels + c3_in_channels + c4_in_channels,
                                      trans_channels=[c1_in_channels,c2_in_channels,c3_in_channels,c4_in_channels])

            self.inj3 = Skip_InjectionMultiSum_Auto_pool(c3_in_channels,c3_in_channels,activations=nn.ReLU6)
            self.inj2 = Skip_InjectionMultiSum_Auto_pool(c2_in_channels,c2_in_channels,activations=nn.ReLU6)
            self.inj1 = Skip_InjectionMultiSum_Auto_pool(c1_in_channels,c1_in_channels,activations=nn.ReLU6)
            self.ffn0 = nn.Sequential(
                CBAMBlock(channel=c4_in_channels,reduction=8, kernel_size=7),
                ConvModule(c4_in_channels, c3_in_channels,k=3,p=1)
            )
            self.ffn1 = nn.Sequential(
                CBAMBlock(channel=c3_in_channels,reduction=4, kernel_size=7),
                ConvModule(c3_in_channels, c2_in_channels,k=3,p=1)
            )
            self.ffn2 = nn.Sequential(
                CBAMBlock(channel=c2_in_channels,reduction=2, kernel_size=7),
                ConvModule(c2_in_channels, c1_in_channels,k=3,p=1)
            )
            self.ffn3 = nn.Sequential(
                CBAMBlock(channel=c1_in_channels,reduction=1, kernel_size=7),
                ConvModule(c1_in_channels, 64,k=3,p=1)
            )
        else:
            self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
            self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
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

        if self.attention == 'backbone_multi-levelv7-ii-1-6' or self.attention == 'backbone_multi-levelv7-ii-1-6-v1' \
                or self.attention == 'backbone_multi-levelv7-ii-1-6-v2':
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
            out_feat = _c
        elif self.attention=='backbone_multi-levelv7-ii-1-6-v1-dam' or self.attention=='backbone_multi-levelv7-ii-1-6-v1-dam-criss':
            c4 = self.dam(c4)
            # 全部上采样到128*128
            lateral_c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            lateral_c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out1 = self.ffn1(torch.cat([lateral_c4, lateral_c3], dim=1))
            out2 = self.ffn2(torch.cat([out1, lateral_c2], dim=1))
            out3 = self.ffn3(torch.cat([out2, c1], dim=1))

            _c4 = self.linear_c4(lateral_c4).permute(0, 2, 1).reshape(n, -1, lateral_c4.shape[2], lateral_c4.shape[3])
            _c3 = self.linear_c3(out1).permute(0, 2, 1).reshape(n, -1, out1.shape[2], out1.shape[3])
            _c2 = self.linear_c2(out2).permute(0, 2, 1).reshape(n, -1, out2.shape[2], out2.shape[3])
            _c1 = self.linear_c1(out3).permute(0, 2, 1).reshape(n, -1, out3.shape[2], out3.shape[3])

            sub1 = _c1 - _c2
            sub2 = _c3 - _c4
            sub = self.ffn_sub(torch.cat([sub1, sub2], dim=1))
            sub = self.cca1(sub)

            _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4, sub], dim=1))
            out_feat = _c
        elif self.attention == 'o1'  or self.attention == 'o1-damcriss':
            c4 = self.dam(c4)
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            c3 = self.ffn1(_c4 + c3)
            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            c2 = self.ffn2(_c3 + c2)
            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c1 = self.ffn3(_c2 + c1)
            out_feat = _c1
        elif self.attention == 'o1-cbam':
            c4 = self.f4_dam(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            c3 = self.ffn3(torch.cat([c3 ,_c4],dim=1))
            c3 = self.f3_dam(c3)
            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            c2 = self.ffn2(torch.cat([c2,_c3],dim=1))
            c2 = self.f2_dam(c2)
            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            c1 = self.ffn1(torch.cat([c1,_c2],dim=1))
            _c1 = self.f1_dam(c1)
            _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
            out_feat = self.ffn(torch.cat([_c1,_c2,_c3,_c4],dim=1))

        elif self.attention == 'o1-fam':
            global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]
            c4 = self.dam(c4)
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            c3 = self.ffn1(_c4 + c3 + global_c3)
            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            c2 = self.ffn2(_c3 + c2 + global_c2)
            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c1 = self.ffn3(_c2 + c1 + global_c1)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj':
            global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]
            c4 = self.dam(c4)
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = _c4 + c3
            _c4 = self.inj3(_c4,global_c3)
            c3 = self.ffn1(_c4)

            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = _c3 + c2
            _c3 = self.inj2(_c3, global_c2)
            c2 = self.ffn2(_c3)

            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = _c2 + c1
            _c2 = self.inj1(_c2,global_c1)
            _c1 = self.ffn3(_c2)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj-skip-all-no':
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = _c4 + c3
            _c3 = self.ffn1(_c4)
            _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = _c3 + c2
            _c2 = self.ffn2(_c3)
            _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = _c2 + c1
            _c1 = self.ffn3(_c2)
            out_feat = _c1
        elif self.attention == 'dec_transpose' or \
                self.attention == 'dec_transpose_decCBAM' or \
                self.attention == 'dec_transpose_DAM' or \
                self.attention == 'dec_transpose_AxialDAM' or \
                self.attention == 'dec_transpose_CCA_dec_CCA' or \
                self.attention == 'dec_transpose_CCA':
            _c4 = self.center(c4)
            _c4 = F.interpolate(_c4, size=c4.size()[2:], mode='bilinear', align_corners=False)
            _c4 = self.dec4(torch.cat([_c4,c4],dim=1))
            _c3 = self.dec3(torch.cat([_c4,c3],dim=1))
            _c2 = self.dec2(torch.cat([_c3,c2],dim=1))
            _c1 = self.dec1(torch.cat([_c2,c1],dim=1))
            out_feat = _c1

        elif self.attention == 'dec_transpose_FAMIFM_CBAM_CCA_SAM':
            _c4 = self.center(c4)
            global_info = self.low_FAM_IFM((c1, c2, c3, c4))
            _c4 = F.interpolate(_c4, size=c4.size()[2:], mode='bilinear', align_corners=False)
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]

            _c4 = self.dec4(torch.cat([_c4,c4],dim=1))

            _c3 = self.dec3(torch.cat([_c4,c3,global_c3],dim=1))
            _c2 = self.dec2(torch.cat([_c3,c2,global_c2],dim=1))
            _c1 = self.dec1(torch.cat([_c2,c1,global_c1],dim=1))
            out_feat = _c1
        elif self.attention == 'dec_transpose_FAMIFM' or \
                self.attention == 'dec_transpose_FAMIFM_DAM' or \
                self.attention == 'dec_transpose_FAMIFM_AxialDAM' or \
                self.attention == 'dec_transpose_FAMIFM_CBAM' or \
                self.attention == 'dec_transpose_FAMIFM_decCCA' or \
                self.attention == 'dec_transpose_FAMIFM_CBAM_DAM' or \
                self.attention == 'dec_transpose_FAMIFM_CBAM_CCA_R2' or \
                self.attention == 'dec_transpose_FAMIFM_center_CBAM_DAM' or \
                self.attention == 'dec_transpose_FAMIFM_CCA' or \
                self.attention == 'dec_transpose_FAMIFM_CBAM_CCA':
            if self.attention == 'dec_transpose_FAMIFM_CBAM_CCA_R2':
                _c4 = self.center(c4)
                _c4 = self.center(_c4)
            else:
                _c4 = self.center(c4)
            if self.attention == 'dec_transpose_FAMIFM_center_CBAM_DAM':
                global_info = self.low_FAM_IFM((c1, c2, c3, _c4))
            else:
                global_info = self.low_FAM_IFM((c1, c2, c3, c4))
            _c4 = F.interpolate(_c4, size=c4.size()[2:], mode='bilinear', align_corners=False)
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]

            _c4 = self.dec4(torch.cat([_c4,c4],dim=1))

            _c3 = self.dec3(torch.cat([_c4,c3,global_c3],dim=1))
            _c2 = self.dec2(torch.cat([_c3,c2,global_c2],dim=1))
            _c1 = self.dec1(torch.cat([_c2,c1,global_c1],dim=1))
            out_feat = _c1
        elif self.attention == 'dec_transpose_FAMIFM_DAM_Inj' or\
                self.attention == 'dec_transpose_FAMIFM_CBAM_Inj' or \
                self.attention == 'dec_transpose_FAMIFM_CBAM_CCA_Inj' or \
                self.attention == 'dec_transpose_FAMIFM_CBAM_DAM_Inj':
            _c4 = self.center(c4)
            _c4 = F.interpolate(_c4, size=c4.size()[2:], mode='bilinear', align_corners=False)

            global_info = self.low_FAM_IFM((c1, c2, c3, c4))
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]
            _c4 = self.dec4(torch.cat([_c4, c4], dim=1))
            # 开始inj
            _c4 = F.interpolate(_c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c3 = self.inj3(_c4, global_c3, c3)
            _c3 = self.dec3(_c3)
            _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c2 = self.inj2(_c3, global_c2, c2)
            _c2 = self.dec2(_c2)
            _c1 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c1 = self.inj1(_c1,global_c1,c1)
            _c1 = self.dec1(_c1)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj-skip' or self.attention == 'o1-fam-inj-cbam-skip':
            global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]
            c4 = self.dam(c4)
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = self.inj3(_c4,global_c3,c3)
            c3 = self.ffn1(_c4)

            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = self.inj2(_c3, global_c2,c2)
            c2 = self.ffn2(_c3)

            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = self.inj1(_c2,global_c1,c1)
            _c1 = self.ffn3(_c2)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj-skip-no-dam':
            global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]

            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = self.inj3(_c4,global_c3,c3)
            c3 = self.ffn1(_c4)

            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = self.inj2(_c3, global_c2,c2)
            c2 = self.ffn2(_c3)

            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = self.inj1(_c2,global_c1,c1)
            _c1 = self.ffn3(_c2)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj-skip-no-FAM-IFM':
            # global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = c1
            global_c2 = c2
            global_c3 = c3
            c4 = self.dam(c4)
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = self.inj3(_c4,global_c3,c3)
            c3 = self.ffn1(_c4)

            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = self.inj2(_c3, global_c2,c2)
            c2 = self.ffn2(_c3)

            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = self.inj1(_c2,global_c1,c1)
            _c1 = self.ffn3(_c2)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj-skip-no-FAM-IFM-inj':
            global_c1 = c1
            global_c2 = c2
            global_c3 = c3
            c4 = self.dam(c4)
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = _c4 + global_c3 + c3
            c3 = self.ffn1(_c4)

            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = _c3 + global_c2 + c2
            c2 = self.ffn2(_c3)

            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = _c2 + global_c1 + c1
            _c1 = self.ffn3(_c2)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj-skip-no-dam-FAM-IFM':
            # global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = c1
            global_c2 = c2
            global_c3 = c3
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = self.inj3(_c4,global_c3,c3)
            c3 = self.ffn1(_c4)

            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = self.inj2(_c3, global_c2,c2)
            c2 = self.ffn2(_c3)

            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = self.inj1(_c2,global_c1,c1)
            _c1 = self.ffn3(_c2)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj-skip-no-inj':
            global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]
            c4 = self.dam(c4)
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = _c4+global_c3+c3
            c3 = self.ffn1(_c4)

            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = _c3+global_c2+c2
            c2 = self.ffn2(_c3)

            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = _c2+global_c1+c1
            _c1 = self.ffn3(_c2)
            out_feat = _c1

        elif self.attention == 'o1-fam-inj-skip-no-dam-inj':
            global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = F.interpolate(global_info[0], size=c1.size()[2:], mode='bilinear', align_corners=False)
            global_c2 = F.interpolate(global_info[1], size=c2.size()[2:], mode='bilinear', align_corners=False)
            global_c3 = global_info[2]
            c4 = self.ffn0(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = _c4+global_c3+c3
            c3 = self.ffn1(_c4)

            _c3 = F.interpolate(c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
            _c3 = _c3+global_c2+c2
            c2 = self.ffn2(_c3)

            _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
            _c2 = _c2+global_c1+c1
            _c1 = self.ffn3(_c2)
            out_feat = _c1
        elif self.attention == 'o1-fam-inj-v1':
            global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = global_info[0]
            global_c2 = global_info[1]
            global_c3 = global_info[2]
            c4 = self.dam(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = self.fuse_3_4(torch.cat([_c4,c3],dim=1))
            _inj3 = self.inj3(_c4,global_c3)
            _inj2 = self.inj2(c2,global_c2)
            _inj1 = self.inj1(c1,global_c1)
            _inj2 = F.interpolate(_inj2, size=_inj1.size()[2:], mode='bilinear', align_corners=False)
            _inj3 = F.interpolate(_inj3, size=_inj1.size()[2:], mode='bilinear', align_corners=False)

            out_feat = self.ffn(torch.cat([_inj1,_inj2,_inj3],dim=1))
            _c1 = _inj1
            _c2 = _inj2
            _c3 = _inj3
        elif self.attention == 'o1-fam-inj-v1-cbam':
            global_info = self.low_FAM_IFM((c1,c2,c3,c4))
            global_c1 = global_info[0]
            global_c2 = global_info[1]
            global_c3 = global_info[2]
            c4 = self.dam(c4)
            _c4 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
            _c4 = self.fuse_3_4(torch.cat([_c4,c3],dim=1))
            _inj3 = self.inj3(_c4,global_c3)
            _inj3 = self.f3_cbam(_inj3)
            _inj2 = self.inj2(c2,global_c2)
            _inj2 = self.f2_cbam(_inj2)
            _inj1 = self.inj1(c1,global_c1)
            _inj1 = self.f1_cbam(_inj1)
            _inj2 = F.interpolate(_inj2, size=_inj1.size()[2:], mode='bilinear', align_corners=False)
            _inj3 = F.interpolate(_inj3, size=_inj1.size()[2:], mode='bilinear', align_corners=False)

            out_feat = self.ffn(torch.cat([_inj1,_inj2,_inj3],dim=1))
            _c1 = _inj1
            _c2 = _inj2
            _c3 = _inj3
        else:
            # if self.attention == 'org':
            _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
            _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

            _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

            _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
            _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

            _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        if self.attention == 'org':
            out_feat = self.dropout(_c)
        # x = self.classifier(out_feat)

        # return out_feat,x
        # return out_feat
        return {"out_feat":out_feat,
                "_c1":_c1,
                "_c2":_c2,
                "_c3":_c3,
                "_c4":_c4,}


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
        self.attention = attention
        if 'org' in attention:
            self.classifier = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1)
        elif 'SAM' in attention:
            sam_checkpoint = "pretrained/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.predictor = SamPredictor(self.sam)
            # Freeze the parameters of sam and predictor
            for param in self.sam.parameters():
                param.requires_grad = False
            # for param in self.predictor.parameters():
            #     param.requires_grad = False
            self.SAM_Conv = nn.Conv2d(kernel_size=1,in_channels=256,out_channels=64)
        else:
            self.classifier = nn.Conv2d(64, num_classes, kernel_size=3,padding=1)
        # self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)



    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        # self.predictor.set_torch_image(inputs,inputs.shape)

        # input_point = np.array([[256, 256]])
        # input_label = np.array([1])
        if 'SAM' in self.attention:
            from segment.segment_anything_main.segment_anything.utils.transforms import ResizeLongestSide
            resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)

            batched_input = [{
                    'image': inputs[i,...],
                    'point_coords': resize_transform.apply_coords_torch(torch.tensor([[H//2,H//2]]), (H, W)),
                    'point_labels': torch.tensor([0,1]),
                    'original_size': (H, W)
                } for i in range(inputs.size(0))]
            # batched_input = [
            #     {
            #         'image': image1,
            #         'boxes': resize_transform.apply_coords(torch.tensor([H//2,H//2]), (H, W)),
            #         'original_size': image1.shape[:2]
            #     },
            #     {
            #         'image': prepare_image(image2, resize_transform, sam),
            #         'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
            #         'original_size': image2.shape[:2]
            #     }
            # ]
            sam_outputs = self.sam(batched_input, multimask_output=False)
            # masks, scores, logits, hs, src, iou_token_out = self.predictor.predict(
            #     point_coords=input_point,
            #     point_labels=input_label,
            #     multimask_output=True,
            # )


        backbone_feats = self.backbone.forward(inputs)
        # out_feat,out_classifier = self.decode_head.forward(backbone_feats)
        decodehead_out = self.decode_head.forward(backbone_feats)
        out_feat = decodehead_out['out_feat']
        if 'SAM' in self.attention:
            src = self.SAM_Conv(sam_outputs['src'])
            out_feat = out_feat + src
        # out_feat = self.reduct4loss(out_feat)
        if self.seghead_last:
            out_feat = F.interpolate(out_feat, size=(H, W), mode='bilinear', align_corners=True)
            x = self.classifier(out_feat)
            out_classifier = x
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






if __name__ == '__main__':
    # ckpt_path = '../../../pretrained/segformer_b2_weights_voc.pth'
    # sd = torch.load(ckpt_path,map_location='cpu')
    from torchsummary import summary
    from thop import profile
    # model = ResSegFormer(num_classes=3, phi='b2',res='resnet34', pretrained=False,version='v2')
    model = SegFormer(num_classes=3, phi='b4', pretrained=False,attention='dec_transpose_FAMIFM_CBAM_CCA')
    input = torch.randn(1,3,256,256)
    model = model.to('cuda')
    input = input.to('cuda')
    summary(model, input_size=(3, 256, 256))
    macs, params = profile(model, inputs=input)
    print(f"MACs (FLOPs): {macs}, Params: {params}")
    # out = model(img)
    # logits = out['out']
    # print(logits.shape)
    # print(logits.shape)