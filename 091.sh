CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translateCEDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/aux1e_backbone_pretrained_flip_rotate_scale_translateCEDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translateCECBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translate_CEContrastPixelCBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translate_CEContrastCorrectPixelCBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translateCEDCBDFCLoss_Criss_Attention_R2_V1

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/backbone_pretrained_flip_rotate_scale_translateCELoss



