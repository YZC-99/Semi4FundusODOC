CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50Mydeeplabv3plusplus/random1_ODOC_backbone_pretrained_flip_rotateDCBDFCLoss
# 20230908
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/segformer/backbone_b2_pretrained_whole_inVOC_flip_rotateDCBDFCLoss_segheadlast
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/segformer/backbone_b2_pretrained_whole_inVOC_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_whole_inVOC_flip_rotateDCBDFCLoss




