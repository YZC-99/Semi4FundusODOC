import torch.nn as nn
import torch
from typing import List,Tuple, Dict, Any, Optional
from segment.modules.semseg.nn import Unet_Encoder,Unet_Decoder,OutConv

class UNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=3,
                 base_c=64,
                 bilinear=True,
                 ):
        super(UNet, self).__init__()
        self.base_c = base_c
        self.bilinear = bilinear
        self.encoder = Unet_Encoder(in_channels,self.base_c,bilinear=bilinear)
        self.decoder = Unet_Decoder(self.base_c,bilinear=bilinear)
        self.out_conv = OutConv(self.base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_dict = self.encoder(x)
        x = self.decoder(x_dict)['x_up4']
        logits = self.out_conv(x)
        return {'out':logits,'backbone_features':x}