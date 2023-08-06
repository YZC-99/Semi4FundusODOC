# done
CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90 &
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_scratch_tgt_prototype

CUDA_VISIBLE_DEVICES=0 python main.py --config refuge_select/cropped/all &
CUDA_VISIBLE_DEVICES=1 python main.py --config refuge_select/cropped/test &
CUDA_VISIBLE_DEVICES=2 python main.py --config refuge_select/cropped/train &
CUDA_VISIBLE_DEVICES=3 python main.py --config refuge_select/cropped/val &
