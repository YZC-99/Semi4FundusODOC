import torch
import torch.nn.functional as F
from torch import nn
from segment.modules.backbone.resnet import resnet18,resnet34, resnet50, resnet101



class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.classifier(dec1)
        out =  F.upsample(final, x.size()[2:], mode='bilinear')


        return {'out':out,
                'out_fuse': dec1,
                'out_classifier':final,
                'backbone_features':center}

class ResUNet(nn.Module):
    def __init__(self, num_classes,bb_pretrained=True,inplace_seven=False):
        super(ResUNet, self).__init__()
        self.backbone = resnet50(pretrained=bb_pretrained, inplace_seven=inplace_seven)

        self.center = _DecoderBlock(2048, 2048, 2048)
        self.dec5 = _DecoderBlock(4096, 2048, 1024)
        self.dec4 = _DecoderBlock(2048, 1024, 512)
        self.dec3 = _DecoderBlock(1024, 512, 256)
        self.dec2 = _DecoderBlock(512, 256, 128)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x(2,3,256,256)
        # c1(2,256,64,64)
        # c2(2,512,32,32)
        # c3(2,1024,32,32)
        # c4(2,2048,32,32)
        c1,c2,c3,c4 = self.backbone.base_forward(x)
        center = self.center(c4)
        dec5 = self.dec5(torch.cat([center, F.upsample(c4, center.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([dec5, F.upsample(c3, dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(c2, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(c1, dec3.size()[2:], mode='bilinear')], 1))
        # dec2 128维度
        dec1 = self.dec1(dec2)
        final = self.classifier(dec1)
        out =  F.upsample(final, x.size()[2:], mode='bilinear')

        return {'out':out,
                'out_fuse': dec1,
                'out_classifier':final,
                'backbone_features':center}

if __name__ == '__main__':
    resunet = ResUNet(num_classes=3,bb_pretrained=False,inplace_seven=False)
    # model = smp.Unet(encoder_name='')
    input = torch.randn(2,3,256,256)
    out = resunet(input)
    logits = out['out']
    print(logits.shape)