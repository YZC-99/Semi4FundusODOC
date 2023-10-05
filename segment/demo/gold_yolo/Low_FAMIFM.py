from segment.demo.gold_yolo.common import AdvPoolFusion, SimFusion_3in, SimFusion_4in
from segment.demo.gold_yolo.reppan import Conv
from segment.demo.gold_yolo.yolov6.layers.common import RepVGGBlock, BottleRep, BepC3, RepBlock, SimConv
import torch
from torch import nn


class FAMIFM(nn.Module):
    def __init__(self, fusion_in = 1152,embed_dim_p = 192,fuse_block_num = 3,trans_channels = [64,256,320,512]):
        super().__init__()
        self.trans_channels = trans_channels
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
            Conv(fusion_in, embed_dim_p, kernel_size=1, stride=1, padding=0),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(trans_channels[0:3]), kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        low_fam_out = self.low_FAM(x)
        low_ifm_out = self.low_IFM(low_fam_out)
        low_global_info = low_ifm_out.split(self.trans_channels[0:3], dim=1)
        return low_global_info
if __name__ == '__main__':

    c1 = torch.randn(2,64,64,64)
    c2 = torch.randn(2,256,32,32)
    c3 = torch.randn(2,320,16,16)
    c4 = torch.randn(2,512,8,8)

    input = (c1,c2,c3,c4)
    famifm = FAMIFM()
    out = famifm(input)
    print(out[0].shape)
    print(out[1].shape)
