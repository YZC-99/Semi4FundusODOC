from segment.modules.semseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F
from segment.modules.nn.dysapmle import DySample
from segment.modules.semseg.nn import ScaledDotProductAttention
from segment.modules.semseg.nn import Attention,CrissCrossAttention,CoordAtt
from segment.modules.semseg.CTF import cft

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
    def __init__(self, backbone, nclass,Isdysample = False,inplace_seven=False,bb_pretrained = False,attention=None,seghead_last=False):
        super(DeepLabV3Plus, self).__init__(backbone,inplace_seven,bb_pretrained)
        self.attention = attention
        self.seghead_last = seghead_last
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
        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)
        if self.attention == 'CrossAttention':
            self.c2_to_c3 = nn.Sequential(nn.Conv2d(c2_level_channels, c3_level_channels, 1, bias=False),
                                        nn.BatchNorm2d(c3_level_channels),
                                        nn.ReLU(True))
            self.diff_to_fuse = nn.Sequential(nn.Conv2d(c3_level_channels, 256, 1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(True))
            self.cross_attention = ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=4)
            self.mlp_diff = nn.Sequential(
                                    nn.Conv1d(128*128,4096,1,bias=False),
                                    nn.ReLU(4096)
            )
            self.mlp_fuse = nn.Sequential(
                                    nn.Conv1d(128*128,4096,1,bias=False),
                                    nn.ReLU(4096)
            )
            self.mlp_fuse_to_high = nn.Sequential(
                                    nn.Conv1d(4096,128*128,1,bias=False),
                                    nn.ReLU(128*128)
            )

        elif self.attention == 'Criss_Attention' or self.attention == 'Criss_Attention_R2_V1' or \
                self.attention == 'Criss_Attention_R2_V2' or \
                self.attention == 'Criss_Attention_R2_V3' \
                or self.attention == 'Criss_Attention_R2':
            self.c2_to_c3 = nn.Sequential(nn.Conv2d(c2_level_channels, c3_level_channels, 1, bias=False),
                                        nn.BatchNorm2d(c3_level_channels),
                                        nn.ReLU(True))
            self.diff_reduc = nn.Sequential(nn.Conv2d(c3_level_channels, 64, 1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True))
            self.criss_cross_attention = CrissCrossAttention(64)
            # self.diff_increase = nn.Sequential(nn.Conv2d(64, c3_level_channels, 1, bias=False),
            #                             nn.BatchNorm2d(c3_level_channels),
            #                             nn.ReLU(True))

            self.fuse_diff_out = nn.Sequential(nn.Conv2d(256 + 64, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))
        elif self.attention == 'Cross_CrissAttention':
            self.c2_to_c3 = nn.Sequential(nn.Conv2d(c2_level_channels, c3_level_channels, 1, bias=False),
                                        nn.BatchNorm2d(c3_level_channels),
                                        nn.ReLU(True))
            self.diff_reduc = nn.Sequential(nn.Conv2d(c3_level_channels, 64, 1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True))
            self.fuse_reduc = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True))
            self.criss_cross_attention1 = CrissCrossAttention(64)
            self.criss_cross_attention2 = CrissCrossAttention(64)
            self.fuse_diff_out = nn.Sequential(nn.Conv2d(256 + 64, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))
        elif self.attention == 'Coordinate_Attention':
            self.c2_to_c3 = nn.Sequential(nn.Conv2d(c2_level_channels, c3_level_channels, 1, bias=False),
                                        nn.BatchNorm2d(c3_level_channels),
                                        nn.ReLU(True))
            # self.diff_reduc = nn.Sequential(nn.Conv2d(c3_level_channels, 64, 1, bias=False),
            #                             nn.BatchNorm2d(64),
            #                             nn.ReLU(True))
            self.coordinate_attention = CoordAtt(c3_level_channels,c3_level_channels,reduction=8)
            self.fuse_diff_out = nn.Sequential(nn.Conv2d(c3_level_channels + 256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))
        elif self.attention == 'Criss_Coordinate_Attention':
            self.c2_to_c3 = nn.Sequential(nn.Conv2d(c2_level_channels, c3_level_channels, 1, bias=False),
                                          nn.BatchNorm2d(c3_level_channels),
                                          nn.ReLU(True))
            self.diff_reduc = nn.Sequential(nn.Conv2d(c3_level_channels, 64, 1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True))
            self.criss_cross_attention = CrissCrossAttention(64)
            self.coordinate_attention = CoordAtt(c3_level_channels, c3_level_channels, reduction=8)
            self.fuse_diff_out = nn.Sequential(nn.Conv2d(c3_level_channels + 256, 256, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(True),

                                               nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(True),
                                               nn.Dropout(0.1, False))

        self.Isdysample = Isdysample
        if self.Isdysample:
            self.dysample = DySample(in_channels=nclass, scale=4,style='lp', groups=3)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        backbone_out = self.backbone.base_forward(x)
        c1, c2, c3, c4 = backbone_out['c1'],backbone_out['c2'],backbone_out['c3'],backbone_out['c4']
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
        if self.attention == 'CrossAttention':
            c2 = self.c2_to_c3(c2)
            diff = c2 - c3
            diff = self.diff_to_fuse(diff)
            diff = F.interpolate(diff, size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
            # out_fuse = out_fuse + diff
            b_fuse,c_fuse,h_fuse,w_fuse = out_fuse.size()
            diff = diff.view(b_fuse,-1,c_fuse)
            out_fuse = out_fuse.view(b_fuse,-1,c_fuse)
            # 这里直接爆显存了 估计会是128*128的长度，现在需要将它降低到4096长
            diff = self.mlp_diff(diff)
            out_fuse_lowlevel = self.mlp_fuse(out_fuse)
            out_fuse_lowlevel = self.cross_attention(diff,out_fuse_lowlevel,out_fuse_lowlevel)
            out_fuse_highlevel = self.mlp_fuse_to_high(out_fuse_lowlevel)
            out_fuse_highlevel = out_fuse_highlevel.view(b_fuse,c_fuse,h_fuse,w_fuse)
            out_fuse = out_fuse.view(b_fuse,c_fuse,h_fuse,w_fuse)

            out_fuse = out_fuse + out_fuse_highlevel
        elif self.attention == 'Criss_Attention':
            c2 = self.c2_to_c3(c2)
            diff = c2 - c3
            diff = self.diff_reduc(diff)
            diff = self.criss_cross_attention(diff)
            # diff = self.diff_increase(diff)
            diff = F.interpolate(diff,size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
            out_fuse = self.fuse_diff_out(torch.cat([out_fuse, diff], dim=1))
        elif self.attention == 'Criss_Attention_R2':
            c2 = self.c2_to_c3(c2)
            diff = c2 - c3
            diff = self.diff_reduc(diff)
            diff = self.criss_cross_attention(diff)
            diff = self.criss_cross_attention(diff)
            diff = F.interpolate(diff, size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
            out_fuse = self.fuse_diff_out(torch.cat([out_fuse, diff], dim=1))
        elif self.attention == 'Criss_Attention_R2_V1':
            # c2 = self.c2_to_c3(c2)
            # diff = c2 - c3
            diff = self.diff_reduc(c3)
            diff = self.criss_cross_attention(diff)
            diff = F.interpolate(diff,size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
            out_fuse = self.fuse_diff_out(torch.cat([out_fuse, diff], dim=1))
        elif self.attention == 'Criss_Attention_R2_V2':
            c2 = self.c2_to_c3(c2)
            diff = c2 + c3
            diff = self.diff_reduc(diff)
            diff = self.criss_cross_attention(diff)
            diff = F.interpolate(diff,size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
            out_fuse = self.fuse_diff_out(torch.cat([out_fuse, diff], dim=1))
        elif self.attention == 'Criss_Attention_R2_V3':
            c2 = self.c2_to_c3(c2)
            # diff = c2 + c3
            diff = self.diff_reduc(c2)
            diff = self.criss_cross_attention(diff)
            diff = F.interpolate(diff,size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
            out_fuse = self.fuse_diff_out(torch.cat([out_fuse, diff], dim=1))
        elif self.attention == 'Cross_CrissAttention':
            c3 = self.diff_reduc(c3)
            c3 = self.criss_cross_attention1(c3)
            out_fuse_shape = out_fuse.shape[-2:]
            # 将out_fuse缩小
            out_fuse = F.interpolate(out_fuse,size=c3.shape[-2:], mode="bilinear", align_corners=True)
            out_fuse_reducted = self.fuse_reduc(out_fuse)
            out_cross_criss_att = self.criss_cross_attention2.cross_forward(c3,out_fuse_reducted)
            out_cross_criss_att = F.interpolate(out_cross_criss_att,size=out_fuse_shape, mode="bilinear", align_corners=True)
            out_fuse = F.interpolate(out_fuse,size=out_fuse_shape, mode="bilinear", align_corners=True)
            out_fuse = self.fuse_diff_out(torch.cat([out_fuse, out_cross_criss_att], dim=1))
        elif self.attention == 'Coordinate_Attention':
            # c2 = self.c2_to_c3(c2)
            # diff = c2 + c3
            # diff = self.diff_reduc(c2)
            diff = self.coordinate_attention(c3)
            diff = F.interpolate(diff, size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
            out_fuse = self.fuse_diff_out(torch.cat([out_fuse, diff], dim=1))

        if self.seghead_last:
            out_classifier = F.interpolate(out_fuse, size=(h, w), mode="bilinear", align_corners=True)
            out = self.classifier(out_classifier)
        else:
            out_classifier = self.classifier(out_fuse)
            if self.Isdysample:
                out = self.dysample(out_classifier)
            else:

                out = F.interpolate(out_classifier, size=(h, w), mode="bilinear", align_corners=True)

        return {'out':out,
                'out_classifier':out_classifier,
                'c3':c3,
                'out_features':out_fuse,
                'backbone_features':backbone_feats}

class My_DeepLabV3PlusPlus(BaseNet):
    def __init__(self, backbone, nclass,inplace_seven=False,bb_pretrained = False,attention=None):
        super(My_DeepLabV3PlusPlus, self).__init__(backbone,inplace_seven,bb_pretrained)
        self.attention = attention

        low_level_channels = self.backbone.channels[0]
        c2_level_channels = self.backbone.channels[1]
        c3_level_channels = self.backbone.channels[2]
        high_level_channels = self.backbone.channels[-1]

        self.lateral_connections_c4 = ASPPModule(high_level_channels, (12, 24, 36), down_ratio=8)
        self.lateral_connections_c3 = ASPPModule(c3_level_channels, (12, 24, 36), down_ratio=4)
        self.lateral_connections_c2 = ASPPModule(c2_level_channels, (12, 24, 36), down_ratio=2)
        self.lateral_connections_c1 = ASPPModule(low_level_channels, (12, 24, 36), down_ratio=1, height_down=2)

        self.cft4 = cft(256,nclass)
        self.cft3 = cft(256,nclass)
        self.cft2 = cft(256,nclass)

        # self.head = ASPPModule(high_level_channels, (12, 24, 36))

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

        self.fuse2 = nn.Sequential(nn.Conv2d(256*4, 512, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(True),

                                  nn.Conv2d(512, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))

        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]
        c1, c2, c3, c4 = self.backbone.base_forward(x)
        c_lateral_3 = self.lateral_connections_c3(c3)
        c_lateral_2 = self.lateral_connections_c2(c2)
        c_lateral_1 = self.lateral_connections_c1(c1)

        backbone_feats = c4
        c4 = self.lateral_connections_c4(c4)
        cft4_out = self.cft4(c4,c_lateral_3)
        cft3_out = self.cft3(cft4_out,c_lateral_2)
        cft2_out = self.cft2(cft3_out,c_lateral_1)

        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(c1)

        out = torch.cat([c1, c4], dim=1)
        # 使用difference与out_fuse做cross_attention
        out_fuse = self.fuse(out)
        cft4_out = F.interpolate(cft4_out, size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
        cft3_out = F.interpolate(cft3_out, size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)
        cft2_out = F.interpolate(cft2_out, size=out_fuse.shape[-2:], mode="bilinear", align_corners=True)

        out_fuse2 = self.fuse2(torch.cat([out_fuse,cft4_out,cft3_out,cft2_out],dim=1))

        # 使用上采样插值也许带来的效果
        out_classifier = self.classifier(out_fuse2)
        out = F.interpolate(out_classifier, size=(h, w), mode="bilinear", align_corners=True)

        return {'out':out,
                'out_classifier':out_classifier,
                'c3':c3,
                'out_features':out_fuse2,
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
    def __init__(self, in_channels, atrous_rates,down_ratio=8, height_down=1):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // down_ratio
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1,stride=height_down, bias=False),
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
    model = DeepLabV3Plus(backbone='resnet50', nclass=3,attention = 'Criss_Attention')
    out = model(input)
