#20230908
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDCBDFCLoss

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDCBDFCLoss_attention_subv1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_flip_rotate_scale_translateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_whole_inVOC_flip_rotate_scale_translateDCBDFCLoss


CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_flip_rotate_scale_translateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_scratch_flip_rotate_scale_translateDCBDFCLoss

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config Drishti-GS/cropped_sup256x256/segformer/backbone_b4_pretrained_flip_rotate_scale_translateDCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config Drishti-GS/cropped_sup256x256/segformer/backbone_b5_pretrained_flip_rotate_scale_translateDCBDFCLoss


CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDCBDFCLoss_attention_subv2

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_flip_rotate_scale_translate_noise_cutoutDCBDFCIoULoss_attention_subv1
