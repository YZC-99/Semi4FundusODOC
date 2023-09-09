# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment.modules.semseg.nn import Attention,CrissCrossAttention,CoordAtt

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
                'backbone_features':backbone_feats}

if __name__ == '__main__':
    # ckpt_path = '../../../pretrained/segformer_b2_weights_voc.pth'
    # sd = torch.load(ckpt_path,map_location='cpu')

    model = SegFormer(num_classes=3, phi='b2', pretrained=False,attention='subv1')
    img = torch.randn(2,3,512,512)
    out = model(img)
    logits = out['out']
    print(logits.shape)