#20230908
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_CEDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/aux1e_backbone_pretrained_flip_rotate_scale_translateCEDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_CECBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_CEContrastPixelCBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_CEContrastCorrectPixelCBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_CEDCBDFCLoss_Criss_Attention_R2_V1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_CELoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_CEIoUABLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translate_CEDCBDFCLoss_Criss_Attention_R2_V1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translate_cutout_CEDCBDFCLoss_Criss_Attention_R2_V1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translate_noise_cutout_CEDCBDFCLoss_Criss_Attention_R2_V1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translate_noise_cutout_blur_CEDCBDFCLoss_Criss_Attention_R2_V1

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translate_CEDCBDFCLoss

#
#REFUGE
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config REFUGE/cropped_sup256x256/segformer/OHEM9e-1_backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDC_IoU_FC1e-1_stop10_Loss_attention_sub_addv3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config RIM-ONE/cropped_sup256x256/segformer/OHEM9e-1_backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDC_IoU_FC1e-1_stop10_Loss_attention_sub_addv3

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/segformer/OHEM9e-1_backbone_b4_pretrained_flip_rotate_scale_translate_noise_cutoutDCBD_FC1e-1Loss_attention_sub_addv3

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/segformerV2/OHEM9e-1_backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDCBD_FC1e-1Loss_attention_sub_addv3


CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/segformerV2/OHEM9e-1_flip_rotate_scale_translate_noise_cutoutDCBD_FC1e-1Loss_attention_sub_addv3


CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/segformerV2/imgnet_normal_OHEM9e-1_backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDCBD_FC1e-1Loss_attention_sub_addv3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/segformerV2/self_normal_OHEM9e-1_backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDCBD_FC1e-1Loss_attention_sub_addv3




