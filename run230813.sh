#CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi10_None_scratch_DCBDLoss
CUDA_VISIBLE_DEVICES=0,1 python main.py  --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi10_B_CJ_CO_RG_DCBDLoss &
CUDA_VISIBLE_DEVICES=2,3 python main.py  --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi10_B_CJ_CO_RG_scratch_DCBDLoss_minusBoundary2
PID1=$!

wait $PID1
/usr/bin/shutdown

