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
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/losses/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_2DC_2BD_Loss_attention-sub-add-v7

# augs
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs200_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_2DC_2BD_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/augs/epochs200_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_noise_CE_2DC_2BD_Loss_attention-sub-add-v3


# -parameters
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/parameters/40decounpling-params_epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/parameters/undecounpling-params_epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v3



#org_segformer
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup10e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_DC_BD_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup10e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_3DC_3BD_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/losses/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_CE_3DC_3BD_Loss
# augs
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/augs/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_flip_rotate_translate_scale_CE_Loss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/org_segformer/epochs100_warmup25e-2_lr1e-4_OHEM9e-1_backbone_b2_pretrained_scale_CE_Loss









