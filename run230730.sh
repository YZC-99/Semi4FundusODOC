#done
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align3e-1_logitsTransform &
#CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/semi/50/ODOC_semi50_align3e-1_logitsTransform

CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align1e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/sup/random1_ODOC_sup_align2e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/sup/random1_ODOC_sup_align4e-1_logitsTransform &
PID1=$!

wait $PID1
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align5e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/sup/random1_ODOC_sup_align6e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/sup/random1_ODOC_sup_align7e-1_logitsTransform &
PID2=$!

wait $PID2
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align8e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/sup/random1_ODOC_sup_align9e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/sup/random1_ODOC_sup_align10e-1_logitsTransform &
PID3=$!


