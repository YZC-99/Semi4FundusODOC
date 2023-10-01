CUDA_VISIBLE_DEVICES=0 python main.py  --lr 3.0e-4 --warmup 0.15 --BD_loss 2.0 --FC_loss 0.1 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config REFUGE/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise --d "学习率" &
sleep 15
CUDA_VISIBLE_DEVICES=1 python main.py  --lr 4.0e-4 --warmup 0.15 --BD_loss 2.0 --FC_loss 0.1 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config REFUGE/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise --d "学习率" &
sleep 15
CUDA_VISIBLE_DEVICES=2 python main.py  --lr 5.0e-4 --warmup 0.15 --BD_loss 2.0 --FC_loss 0.1 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config REFUGE/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise --d "学习率"