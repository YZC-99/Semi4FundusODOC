import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from segment.modules.backbone.resnet import resnet18,resnet34, resnet50, resnet101

# 6.9定稿版本
# 参考：
# arxiv 1505.04597
# arxiv 1801.05746，官方实现：https://github.com/ternaus/TernausNet
# https://blog.csdn.net/github_36923418/article/details/83273107
# pixelshuffle参考: arxiv 1609.05158

backbone = 'resnet50'


class DecoderBlock(nn.Module):
    """
    U-Net中的解码模块

    采用每个模块一个stride为1的3*3卷积加一个上采样层的形式

    上采样层可使用'deconv'、'pixelshuffle', 其中pixelshuffle必须要mid_channels=4*out_channles

    定稿采用pixelshuffle

    BN_enable控制是否存在BN，定稿设置为True
    """

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1,
                              bias=False)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,

                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = self.relu2(x)
        return x

class backboneModule(nn.Module):
    """
    定稿使用resnet50作为backbone

    BN_enable控制是否存在BN，定稿设置为True
    """
    def __init__(self,backbone = 'resnet50', resnet_pretrain=False):
        super().__init__()
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=resnet_pretrain)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=resnet_pretrain)

        self.firstconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
    def forward(self,x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        return  e1,e2,e3,x


class Resnet_Unet(nn.Module):
    """
    定稿使用resnet50作为backbone

    BN_enable控制是否存在BN，定稿设置为True
    """

    def __init__(self,num_classes = 3, BN_enable=True, resnet_pretrain=False):
        super().__init__()
        self.BN_enable = BN_enable
        # encoder部分
        # 使用resnet34或50预定义模型，由于单通道入，因此自定义第一个conv层，同时去掉原fc层
        # 剩余网络各部分依次继承
        # 经过测试encoder取三层效果比四层更佳，因此降采样、升采样各取4次
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=resnet_pretrain)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=resnet_pretrain)
            filters = [64, 256, 512, 1024, 2048]

        self.backbone = backboneModule(backbone,resnet_pretrain=resnet_pretrain)

        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable)
        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        e1,e2,e3,x = self.backbone(x)

        center = self.center(e3)

        d2 = self.decoder1(torch.cat([center, e2], dim=1))
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))
        d4 = self.decoder3(torch.cat([d3, x], dim=1))
        out = self.final(d4)

        return {'out':out,
                'out_feat':d4,
                'backbone_features':e3}


class my_resnet_unet(nn.Module):
    def __init__(self,num_classes = 3, resnet_pretrain=False,BN_enable=True):
        super().__init__()
        self.BN_enable = BN_enable
        self.backbone = resnet50(pretrained=resnet_pretrain)
        filters = [64, 256, 512, 1024, 2048]
        self.center = DecoderBlock(in_channels=filters[4], mid_channels=filters[4] * 4, out_channels=filters[4],
                                   BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[4] + filters[3], mid_channels=filters[3] * 4,
                                     out_channels=filters[3], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable)
        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
                nn.Sigmoid()
            )
    def forward(self,x):
        backbone_out = self.backbone.base_forward(x)
        x_relu,x, c1, c2, c3,c4 = backbone_out['x_relu'],backbone_out['x'],backbone_out['c1'],backbone_out['c2'],backbone_out['c3'],backbone_out['c4']
        center = self.center(c4)

        c3 = F.interpolate(c3, size=center.shape[-2:], mode="bilinear", align_corners=True)
        d2 = self.decoder1(torch.cat([center, c3], dim=1))

        c2 = F.interpolate(c2, size=d2.shape[-2:], mode="bilinear", align_corners=True)
        d3 = self.decoder2(torch.cat([d2, c2], dim=1))


        x_relu = F.interpolate(x_relu, size=d3.shape[-2:], mode="bilinear", align_corners=True)
        d4 = self.decoder3(torch.cat([d3, x_relu], dim=1))
        out = self.final(d4)
        return {'out':out,
                'out_feat':c3,
                'backbone_features':c3}

if __name__ == '__main__':
    img = torch.randn(2,3,512,512)
    model = my_resnet_unet(num_classes=3,resnet_pretrain=False)
    out = model(img)
    print(out.shape)