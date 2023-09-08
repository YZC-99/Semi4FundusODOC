CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_flip_rotate_scale_translateDCBDFCLoss

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_pretrained_whole_inVOC_flip_rotate_scale_translateDCBDFCLoss

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 --config Drishti-GS/cropped_sup256x256/segformer/backbone_b2_scratch_flip_rotate_scale_translateDCBDFCLoss
