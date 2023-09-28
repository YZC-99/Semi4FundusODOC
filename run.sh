CUDA_VISIBLE_DEVICES=0 python main.py --lr 10e-4 --backbone b4 --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
CUDA_VISIBLE_DEVICES=1 python main.py --lr 9e-4 --backbone b4 --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
CUDA_VISIBLE_DEVICES=2 python main.py --lr 8e-4 --backbone b4 --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
CUDA_VISIBLE_DEVICES=3 python main.py --lr 7e-4 --backbone b4 --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
