
# v7-ii-1-3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/backbone_b2_pretrained_flip_rotate_translate_scale_CE_Loss

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/backbone_b2_pretrained_flip_rotate_translate_scale_CE_5e-1ContrastCrossPixelCorrect_Loss

#-loss
#--CEpair
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/CEpair/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_1e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/CEpair/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_2e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/CEpair/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_3e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/CEpair/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_4e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/CEpair/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_5e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/CEpair/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_6e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/CEpair/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_7e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/CEpair/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_8e-1FC_Loss

#--contrast
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_1e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_2e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_3e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_4e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_5e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_6e-1Contrast_Loss

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_7e-1Contrast_Loss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_8e-1Contrast_Loss

#--dice
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Dice/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1eDC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Dice/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Dice/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Dice/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eDC_2e-1CEpair_5e-1ContrastCrossPixelCorrect_Loss
#--iou
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/IoU/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1eIoU_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/IoU/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eIoU_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/IoU/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1eIoU_2e-1CEpair_Loss

#--focal
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_3e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1eFC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_5e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_4e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_6e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_7e-1FC_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_1e-1CEpair_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_2e-1CEpair_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_3e-1CEpair_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_4e-1CEpair_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_5e-1CEpair_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_6e-1CEpair_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_7e-1CEpair_Loss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-3/loss/Focal/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_8e-1CEpair_Loss



