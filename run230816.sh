CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_sup05_DCBDFCLoss
CUDA_VISIBLE_DEVICES=2,3 python main.py  --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_sup02_DCBDFCLoss

