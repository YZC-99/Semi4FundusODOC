# done
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_DCBDFCLoss
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEIOUABLLoss
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CECBLLoss



#doing
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_dySampleCEContrastCorrectPixelCBLLoss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEContrastCenterCBLLoss &
CUDA_VISIBLE_DEVICES=2 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEPairwiseCBLLoss &


