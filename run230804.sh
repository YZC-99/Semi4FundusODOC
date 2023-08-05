CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup256x256/random1_ODOC_sup
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup256x256/random1_ODOC_sup_align3e-1_logitsTransform
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup30
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup40
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup50
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup60
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup70
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup80
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup90

#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup &
#PID1=$!
#wait $PID1
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup_align3e-1_logitsTransform &
#PID2=$!
#wait $PID2
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup/random1_ODOC_sup_blv &
#PID3=$!
#wait $PID3



