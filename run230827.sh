# done
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_DCBDFCLoss
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEIOUABLLoss
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CECBLLoss



#doing in:内蒙A区 / 093机 c6d2118f3c-09af340d
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_dySampleCEContrastCorrectPixelCBLLoss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEContrastCenterCBLLoss &
CUDA_VISIBLE_DEVICES=2 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CEPairwiseCBLLoss &

#doing in 内蒙A区 / 091机 c6d211b23c-41818353
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_CECBLwoContextLoss


