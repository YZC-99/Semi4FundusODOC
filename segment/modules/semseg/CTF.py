import torch
import torch.nn as nn
import torch.nn.functional as F
from segment.modules.semseg.nn import CrissCrossAttention


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
    def __init__(self, in_channels, atrous_rates,down_ratio=8,height_down = 1):
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

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class cft(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(cft, self).__init__()
        self.num_classes = num_classes
        self.norm1 = nn.BatchNorm2d(256)

        self.conv = nn.Conv2d(256,num_classes,kernel_size=1,stride=1,bias=False)
        self.alignment = nn.Conv2d(256,256,kernel_size=1,stride=1,bias=False)
        self.feat_reduction = nn.Conv2d(in_channels * num_classes,in_channels,kernel_size=1,stride=1,bias=False)

        self.cross_criss_attention = CrissCrossAttention(256)
        self.norm2 = nn.BatchNorm2d(256)
        self.ffn = MixFeedForward(dim=256,expansion_factor=4)


    def forward(self,x,prior_x):
        B,C,H,W =  x.size()
        x = self.norm1(x)
        mask = self.conv(x)
        mask = mask.view(B,self.num_classes,-1) #(B,C,N)
        mask = F.softmax(mask,dim=1) #(B,C,N)
        feat = self.alignment(x)
        feat = feat.view(B,C,-1)#(B,L,N)
        J_feat = torch.einsum("BCN,BLN->BLCN",feat,mask).reshape(B,-1,H,W)
        # 将J_feat降维
        J_feat = self.feat_reduction(J_feat)
        # 使用J_feat与prior_x进行attention
        criss_attention_out= self.cross_criss_attention.cross_forward(prior_x, J_feat)
        prior_x = criss_attention_out
        prior_x = self.norm2(prior_x)
        prior_F = self.ffn(prior_x)
        prior_F = prior_F + prior_x
        return prior_F


if __name__ == '__main__':
    num_classes = 3
    c4 = torch.randn(2,2048,128,128)
    c3 = torch.randn(2,1024,128,128)
    c2 = torch.randn(2,512,128,128)
    c1 = torch.randn(2,256,256,256)

    lateral_connections_c4 = ASPPModule(2048,(12,24,36),down_ratio=8)
    lateral_connections_c3 = ASPPModule(1024,(12,24,36),down_ratio=4)
    lateral_connections_c2 = ASPPModule(512,(12,24,36),down_ratio=2)
    lateral_connections_c1 = ASPPModule(256,(12,24,36),down_ratio=1,height_down=2)
    c4 = lateral_connections_c4(c4)#(B,256,128,128)
    c3 = lateral_connections_c3(c3)#(B,256,128,128)
    c2 = lateral_connections_c2(c2)#(B,256,128,128)
    c1 = lateral_connections_c1(c1)#(B,256,256,256)
    # print(c4.size())
    # print(c3.size())
    # print(c2.size())
    print(c1.size())

    cft_net4 = cft(256,num_classes)
    cft_net3 = cft(256,num_classes)
    cft_net_out4 = cft_net4(c4,c3)
    cft_net_out3 = cft_net4(cft_net_out4,c2)

    print(cft_net_out4.shape)
    print(cft_net_out3.shape)
