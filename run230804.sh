CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup256x256/random1_ODOC_sup &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup256x256/random1_ODOC_sup_align3e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup256x256/random1_ODOC_sup_blvLoss &