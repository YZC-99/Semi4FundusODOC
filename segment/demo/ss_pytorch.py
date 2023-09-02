from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    weights="/",
    fc_dim=2048)

net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    weights="/",
    fc_dim=2048,
    num_class=2,
    use_softmax=True)
segmentation_module = SegmentationModule(net_encoder,net_decoder)
# print(segmentation_module)
# weights = "./"
# pretrained = True if len(weights) == 0 else False
# print(pretrained)