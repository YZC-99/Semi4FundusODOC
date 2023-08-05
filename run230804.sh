CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup_align3e-1_logitsTransform

