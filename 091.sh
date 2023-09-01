CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_DCBDFCLoss &
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_CEIoUABLLoss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_DCBDFCLoss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_CEIoUABLLoss &
PID1=$!
wait PID1
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_ContrastPixelCBLLoss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_ContrastPixelCBLLoss &
PID2=$!
wait PID2
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_dySampleCEContrastCorrectPixelCBLLoss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_dySampleCEContrastCorrectPixelCBLLoss &
PID3=$!
wait PID3
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_CrossContrastPixelCBLLoss &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_CrossContrastPixelCBLLoss &
PID4=$!
wait PID4
#-------------------------
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_DCBDFCLoss #11500
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_CEIoUABLLoss #11600
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_ContrastPixelCBLLoss # 21926
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config REFUGE/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_dySampleCEContrastCorrectPixelCBLLoss #21926



#CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_DCBDFCLoss #11500
#CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_CEIoUABLLoss #11600
#CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_ContrastPixelCBLLoss # 21926
#CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config Drishti-GS/cropped_sup512x512/res50deeplabv3plus/random1_ODOC_dySampleCEContrastCorrectPixelCBLLoss #21926
