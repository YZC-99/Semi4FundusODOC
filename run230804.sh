# done
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup_align3e-1_logitsTransform &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup_blvLoss &


#
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup90 &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup80 &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup70 &


CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup60 &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup50 &
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_sup512x512/res34deeplabv3plus/random1_ODOC_sup40 &

