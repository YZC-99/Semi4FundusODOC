CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/epochs100_warmup25e-2_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup10e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_DC_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup10e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup10e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_3DC_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/epochs100_start-0-warmup-10e-2_lr-1e-4OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/CBL/epochs100_warmup25e-2__lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Contrast_Correct_Pixel_CBL_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/CBL/epochs100_warmup25e-2__lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_CBL_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/backbone/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b1_pretrained_flip_rotate_translate_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/backbone/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b3_pretrained_flip_rotate_translate_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/backbone/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b4_pretrained_flip_rotate_translate_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v8
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/epochs50_warmup10e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b4_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Contrast_Correct_Pixel_CBL_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Contrast_Correct_Cross_Pixel_CBL_Loss_attention-sub-add-v3

#my_segformer
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v4
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3-1
#-loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_DC_BD_FC1e-1_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_DC_BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_DC_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_3DC_3BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_2IoU_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_FC1e-1-stop20_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_FC1e-1_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v8
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-backbone_subv1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Contrast-Correct-Pixel-CBL-Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_1eContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2IoU_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_5e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_IoU_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_1e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_1e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_1e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_3BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_increase-10e-2_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_3DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Contrast-Correct-Pixel-CBL-Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_1e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_1e-1FC_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_5e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_1eA2C-pair_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_1eA2C-pair_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_1eA2C-pair_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eA2C-pair_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2eA2C-pair_1e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_1e-1ContrastCrossPixelCorrect-start45epoch_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_1e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_5e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_10e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2IoU_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3

#    contrast weight

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_1e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_3e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_4e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_5e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_6e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_7e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_8e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_9e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_10e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_11e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_12e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_13e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_14e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_15e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_16e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_17e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_18e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_19e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/contrast_weight/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_20e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_10e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
# lr
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/lr/epochs100_warmup25e-2_lr1e-4_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/lr/epochs100_warmup25e-2_lr5e-4_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/lr/epochs100_warmup25e-2_lr1e-5_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/lr/epochs100_warmup25e-2_lr1e-3_2cycle_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/lr/epochs100_warmup25e-2_lr1e-3_8cycle_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/lr/epochs100_warmup25e-2_lr1e-3_8cycle-gamma9e-1_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/lr/epochs100_warmup25e-2_lr1e-3-gamma9e-1_2cycle_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3

# seed
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 0  --config Drishti-GS/cropped_sup256x256/my_segformer/seed/seed-0_epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/seed/seed-42_epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 3407  --config Drishti-GS/cropped_sup256x256/my_segformer/seed/seed-3407_epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3

# attentions
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v8
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-backbone-sub-v2
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-multi-addv1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_Loss_attention-backbone-sub-v2
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v2
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v4
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v5
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v6
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_no
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr5e-4_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v4
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr5e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v4

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/attentions/backbone_multi_level/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_v7


# augs
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs200_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs200_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_noise_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup10e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_hvflip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup10e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale5e-1-2_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_all5e-1p_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs200_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_noise_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_noise_cutout_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup10e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3


# sota
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_IoU_5e-1ContrastCrossPixelCorrect_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3

# alf
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config Drishti-GS/cropped_sup256x256/my_segformer/alf/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42 -tune -alf --config Drishti-GS/cropped_sup256x256/my_segformer/alf/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3

# backbones:
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/backbones/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b5_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/backbones/epochs100_warmup25e-2_lr1e-3_OHEM5e-1_backbone_b3_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v3


# -parameters
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/parameters/40decounpling-params_epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/parameters/undecounpling-params_epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v3

# dual_backbone
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/dual_backbone/OHEM5e-1_warmup25e-1_lr-1e-3_backbone_b2_pretrained_flip_rotate_translate_scale_CE-2DC-2BD_Loss-V3

#org_segformer
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup10e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_DC_BD_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup10e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_3DC_3BD_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_3DC_3BD_Loss
# augs
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/augs/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_scale_CE_Loss









