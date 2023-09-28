(
    CUDA_VISIBLE_DEVICES=0 python main.py --lr 6e-5 --warmup 1e-2 --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
    sleep 5  # 延迟1秒
    CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-3 --warmup 1e-2 --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
)
