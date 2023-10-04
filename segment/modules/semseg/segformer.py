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
from segment.demo.DAM.dam import DAM


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
        # if self.attention == 'org':
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])


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

    # model = ResSegFormer(num_classes=3, phi='b2',res='resnet34', pretrained=False,version='v2')
    model = SegFormer(num_classes=3, phi='b2', pretrained=False,attention='backbone_multi-levelv7-ii-1-6-v1-dam-criss')
    img = torch.randn(2,3,256,256)
    out = model(img)
    logits = out['out']
    # print(logits.shape)