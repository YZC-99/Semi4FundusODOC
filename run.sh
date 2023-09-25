CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Dice/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1eDC_Loss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Dice/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_Loss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/IoU/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1eIoU_Loss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/IoU/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eIoU_Loss &
