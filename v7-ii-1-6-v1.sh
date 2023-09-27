
# loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/backbone_b2_pretrained_flip_rotate_translate_scale_CE
# ce_pair

CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_3e-1CEpair_3e-1Contrast_CE


#  only cepair
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_1e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_2e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_3e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_4e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_5e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_6e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_7e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_8e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_9e-1CEpair_CE
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/only_cepair/backbone_b2_pretrained_flip_rotate_translate_scale_10e-1CEpair_CE
# contrast
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_1e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_2e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_3e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_4e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_5e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_7e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_8e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_9e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1FC_10e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_6e-1Contrast_Loss
# only-contrast
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_1e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_3e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_4e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_5e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_7e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_8e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_9e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/contrast/only_contrast/backbone_b2_pretrained_flip_rotate_translate_scale_CE_10e-1Contrast_Loss

# cepair
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_1e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_2e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_3e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_4e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_5e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_6e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_7e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_8e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_9e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss
CUDA_VISIBLE_DEVICES=1 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/ce_pair/backbone_b2_pretrained_flip_rotate_translate_scale_10e-1CEpair_CE_2e-1FC_6e-1Contrast_Loss

#  Dice
CUDA_VISIBLE_DEVICES=0 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/Dice/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_loss
CUDA_VISIBLE_DEVICES=0 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/Dice/backbone_b2_pretrained_flip_rotate_translate_scale_CE_DC_loss

# IoU
CUDA_VISIBLE_DEVICES=0 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/IoU/backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU
CUDA_VISIBLE_DEVICES=0 python main.py  --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/loss/IoU/backbone_b2_pretrained_flip_rotate_translate_scale_CE_2IoU
