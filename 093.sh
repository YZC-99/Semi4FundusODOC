CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup256x256/random1_ODOC_dySampleCEContrastCorrectPixelCBLLoss
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config REFUGE/cropped_sup256x256/random1_ODOC_CrossContrastPixelCBLLoss
