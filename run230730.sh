#
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align3e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/semi/50/ODOC_semi50_align3e-1_logitsTransform &
PID1=$!
wait $PID1
/usr/bin/shutdown


