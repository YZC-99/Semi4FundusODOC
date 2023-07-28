## SEG
#python main.py --config SEG/semi/10/ODOC_semi10
#python main.py --config SEG/semi/10/ODOC_semi20
#python main.py --config SEG/semi/10/ODOC_semi30
#python main.py --config SEG/semi/10/ODOC_semi40
#python main.py --config SEG/semi/10/ODOC_semi50
#python main.py --config SEG/semi/10/ODOC_semi60
#python main.py --config SEG/semi/10/ODOC_semi70
#python main.py --config SEG/semi/10/ODOC_semi80
#python main.py --config SEG/semi/10/ODOC_semi90
#
#
##sup align
#CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/sup/random1_ODOC_sup_align3e-1

CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align1e-1 &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/sup/random1_ODOC_sup_align2e-1 &
CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/sup/random1_ODOC_sup_align3e-1 &
PID1=$!

wait $PID1
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align4e-1 &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/sup/random1_ODOC_sup_align5e-1 &
CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/sup/random1_ODOC_sup_align6e-1 &
PID2=$!

wait $PID2
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align7e-1 &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/sup/random1_ODOC_sup_align8e-1 &
CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/sup/random1_ODOC_sup_align9e-1 &
PID3=$!



