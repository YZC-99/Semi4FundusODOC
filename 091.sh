CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_DCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_CEIoUABLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_ContrastPixelCBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_CrossContrastPixelCBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_dySampleCEContrastCorrectPixelCBLLoss


#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 -v 2 --config REFUGE/cropped_sup256x256/unet/random1_ODOC_DCBDFCLoss
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup256x256/unet/random1_ODOC_DCBDFCLoss
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/random1_ODOC_DCBDFCLoss
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup256x256/res50deeplabv3plus/random1_ODOC_DCLoss
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_DCLoss