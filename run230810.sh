CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_None_DCBDLoss
CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_None_scratch_DCBDLoss
CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_semi90_None_DCBDLoss
