CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/random1_ODOC_DCBDFCLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/random1_ODOC_CEIoUABLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/random1_ODOC_ContrastPixelCBLLoss
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/random1_ODOC_dySampleCEContrastCorrectPixelCBLLoss
