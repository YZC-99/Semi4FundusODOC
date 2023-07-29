# done
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_ODOC_sup_align10e-1 &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/sup/random1_ODOC_sup_align8e-1 &
CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/semi/50/ODOC_semi50_align10e-1 &

#
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/Dual_random1_ODOC_sup

#
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/sup/random1_OD_sup &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/sup/random1_OC_sup

