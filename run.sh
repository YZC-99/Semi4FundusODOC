(
    CUDA_VISIBLE_DEVICES=0 python main.py --lr 6e-5 --warmup 1e-2 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
    sleep 5  # 延迟1秒
    CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-3 --warmup 1e-2 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
)
#-- backbone
#-- epochs

#!/bin/bash

# 循环执行100次
for i in {1..25}
do
    (
    CUDA_VISIBLE_DEVICES=0 python main.py -s i + 0 --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise &
    sleep 5  # 延迟1秒
    CUDA_VISIBLE_DEVICES=1 python main.py -s i + 1 --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise &
    sleep 5  # 延迟1秒
    CUDA_VISIBLE_DEVICES=2 python main.py -s i + 2 --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise &
    sleep 5  # 延迟1秒
    CUDA_VISIBLE_DEVICES=3 python main.py -s i + 3 --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise
    )
    wait  # 等待当前循环内的所有命令完成
done
