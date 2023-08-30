#CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_sup512x512/random1_ODOC_sup10_DCBDFCLoss
#CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEIOUABLLoss
#CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_sup512x512/random1_ODOC_sup10_CECBLLoss
#CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEPairwiseCBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEContrastPixelCBLLoss
