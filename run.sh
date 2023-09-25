CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1e-1FC_Loss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_3e-1FC_Loss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1eFC_Loss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_5e-1FC_Loss &
