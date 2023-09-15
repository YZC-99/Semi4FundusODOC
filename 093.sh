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

CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -s 42  --config Drishti-GS/cropped_sup256x256/my_segformer/epochs100_warmup25e-2_lr1e-3_OHEM9e-1_backbone_b4_pretrained_flip_rotate_translate_CE_Loss_attention-sub-add-v3









