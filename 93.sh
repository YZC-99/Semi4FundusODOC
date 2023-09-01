CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup256x256/random1_ODOC_DCBDFCLoss &
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup256x256/random1_ODOC_CEIoUABLLoss

CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config REFUGE/cropped_sup256x256/random1_ODOC_ContrastPixelCBLLoss &
CUDA_VISIBLE_DEVICES=2 python main.py -ng 1 --config REFUGE/cropped_sup256x256/random1_ODOC_dySampleCEContrastCorrectPixelCBLLoss &
CUDA_VISIBLE_DEVICES=3 python main.py -ng 1 --config REFUGE/cropped_sup256x256/random1_ODOC_CrossContrastPixelCBLLoss &
