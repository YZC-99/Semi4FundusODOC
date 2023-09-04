from segment.modules.semseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F
from segment.modules.nn.dysapmle import DySample
from segment.modules.semseg.nn import ScaledDotProductAttention
from segment.modules.semseg.nn import Attention

class DualDeepLabV3Plus(BaseNet):
    def __init__(self, backbone, nclass,inplace_seven):
        super(DualDeepLabV3Plus, self).__init__(backbone,inplace_seven)

        low_level_channels = self.backbone.channels[0]
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule(high_level_channels, (12, 24, 36))

        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))



        self.classifier1 = nn.Conv2d(256, nclass, 1, bias=True)
        self.classifier2 = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        c1, c2, c3, c4 = self.backbone.base_forward(x)
        backbone_feats = c4
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out)

        out1 = self.classifier1(out)
        out2 = self.classifier2(out)


        out1 = F.interpolate(out1, size=(h, w), mode="bilinear", align_corners=True)
        out2 = F.interpolate(out2, size=(h, w), mode="bilinear", align_corners=True)

        return {'out1':out1,
                'out2':out2,
                'backbone_features':backbone_feats}



class DeepLabV3Plus(BaseNet):
    def __init__(self, backbone, nclass,Isdysample = False,inplace_seven=False,bb_pretrained = False,ca=False):
        super(DeepLabV3Plus, self).__init__(backbone,inplace_seven,bb_pretrained)
        self.ca = ca

        low_level_channels = self.backbone.channels[0]
        c2_level_channels = self.backbone.channels[1]
        c3_level_channels = self.backbone.channels[2]
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule(high_level_channels, (12, 24, 36))

        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))
        if self.ca:
            self.c2_to_c3 = nn.Sequential(nn.Conv2d(c2_level_channels, c3_level_channels, 1, bias=False),
                                        nn.BatchNorm2d(c3_level_channels),
                                        nn.ReLU(True))
            self.diff_to_fuse = nn.Sequential(nn.Conv2d(c3_level_channels, 256, 1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(True))
            self.cross_attention = ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=1)

        # self.cross_attention = ScaledDotProductAttention(d_model=c2, d_k=c1, d_v=c1, h=2)

        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)
        self.Isdysample = Isdysample
        if self.Isdysample:
            self.dysample = DySample(in_channels=nclass, scale=4,style='lp', groups=3)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        c1, c2, c3, c4 = self.backbone.base_forward(x)
        backbone_feats = c4
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)


        out = torch.cat([c1, c4], dim=1)
        # 使用difference与out_fuse做cross_attention
        out_fuse = self.fuse(out)

        # 做c2和c3的差，c2(2,512,64,64)  c3(2,1024,64,64)
        # 将c2送入一个conv，变成 c2(2,1024,64,64)
        # c2-c3:(2,1024,64,64)
        #
        # 将c2-c3的差引入out_fuse做cross attention
        # out_fuse(2,256,128,128)
        # 将c2-c3送入conv，变成(2,256,64,64),然后插值，变成(2,256,128,128)
        if self.ca:
            c2 = self.c2_to_c3(c2)
            diff = c2 - c3
            diff = self.diff_to_fuse(diff)
            diff = F.interpolate(diff, size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
            out_fuse = out_fuse + diff
            b,c,h,w = out_fuse.size()
            diff = diff.view(b,-1,c)
            out_fuse = out_fuse.view(b,-1,c)
            out_fuse = self.cross_attention(diff,out_fuse,out_fuse)
            out_fuse = out_fuse.view(b,c,h,w)

        out_classifier = self.classifier(out_fuse)
        if self.Isdysample:
            out = self.dysample(out_classifier)
        else:
            out = F.interpolate(out_classifier, size=(h, w), mode="bilinear", align_corners=True)

        return {'out':out,
                'out_classifier':out_classifier,
                'c3':c3,
                'out_fuse':out_fuse,
                'backbone_features':backbone_feats}


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

if __name__ == '__main__':
    input  = torch.randn(2,3,512,512)
    model = DeepLabV3Plus(backbone='resnet50', nclass=3)
    out = model(input)
