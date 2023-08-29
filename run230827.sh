#CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_sup512x512/random1_ODOC_sup10_DCBDFCLoss
#CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEIOUABLLoss
CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_sup512x512/random1_ODOC_sup10_CECBLLoss
