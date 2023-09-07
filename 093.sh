# 新指标计算
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_flip_rotate_translate_noiseDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_flip_rotate_translate_noise_randscaleDCBDFCLoss

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Criss_attentionR2_V1_flip_rotateDCBDFCLoss_segheadlast
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_flip_rotate_translate_noise_randscale_cutoutDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_flip_DCBDFCrotateCrossContrastpixelCBLLossV3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup512x512/unet/random1_ODOC_backbone_pretrained_flip_rotate_scale_cutoutDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup512x512/unet/random1_ODOC_backbone_pretrained_flip_rotate_scaleDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup512x512/segformer/random1_ODOC_backbone_b5_pretrained_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/segformer/random1_ODOC_backbone_b5_pretrained_flip_rotateDCBDFCLoss








#20230906
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Criss_Attention_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Criss_AttentionR2_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Criss_AttentionR2_V1_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Criss_AttentionR2_V2_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Criss_AttentionR2_V3_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Criss_AttentionR2_V1_flip_rotateDCBDLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Coordinate_attention_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Cross_Criss_attention_flip_rotateDCBDFCLoss



CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -abs -alf -s 42 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_flip_rotateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_flip_rotateDCBDFCLoss

#
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_flip_rotateDCBDFCLoss_segheadLast
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_backbone_pretrained_Criss_attentionR2_V1_flip_rotateDCBDFCLoss_segheadlast


