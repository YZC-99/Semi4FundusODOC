CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup &
PID1=$!
wait $PID1
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup_align3e-1_logitsTransform &
PID2=$!
wait $PID2
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup_blv &
PID3=$!
wait $PID3
#/usr/bin/shutdown

CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup_align3e-1_logitsTransform-V2
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup_align3e-1_logitsTransform-V3

