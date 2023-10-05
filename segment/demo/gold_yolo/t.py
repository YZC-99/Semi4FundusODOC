from segment.demo.gold_yolo.common import AdvPoolFusion, SimFusion_3in, SimFusion_4in
from segment.demo.gold_yolo.reppan import Conv
from yolov6.layers.common import RepVGGBlock, BottleRep, BepC3, RepBlock, SimConv
import torch
from torch import nn

fusion_in = 1152
embed_dim_p = 192
fuse_block_num = 3
trans_channels = [64,256,320,512]
low_FAM = SimFusion_4in()
low_IFM = nn.Sequential(
            Conv(fusion_in, embed_dim_p, kernel_size=1, stride=1, padding=0),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(trans_channels[0:3]), kernel_size=1, stride=1, padding=0),
        )



c1 = torch.randn(2,64,64,64)
c2 = torch.randn(2,256,32,32)
c3 = torch.randn(2,320,16,16)
c4 = torch.randn(2,512,8,8)

input = (c1,c2,c3,c4)
low_fam_out = low_FAM(input)
low_ifm_out = low_IFM(low_fam_out)
low_global_info = low_ifm_out.split(trans_channels[0:3],dim = 1)
print(low_fam_out.shape)
print(low_ifm_out.shape)
print(low_global_info[0].shape)
