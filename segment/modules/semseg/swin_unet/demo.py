from vision_transformer import  SwinUnet
from config import get_config
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str,default='./swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
args = parser.parse_args()

config = get_config(args)
net = SwinUnet(config,img_size=224,num_classes=3)
input = torch.randn(4,3,224,224)
out = net(input)
print(out.shape)