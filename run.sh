# 循环执行100次
for i in {0..100}
do
    (
    CUDA_VISIBLE_DEVICES=0 python main.py -s $((i + 0)) --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise &
    sleep 60  # 延迟1秒
    CUDA_VISIBLE_DEVICES=1 python main.py -s $((i + 1)) --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise &
    sleep 60  # 延迟1秒
    CUDA_VISIBLE_DEVICES=2 python main.py -s $((i + 2)) --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise &
    )
    wait  # 等待当前循环内的所有命令完成
done
