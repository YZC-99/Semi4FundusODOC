CUDA_VISIBLE_DEVICES=0 main.py  --lr 1.0e-3 --warmup 0.25 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4
--epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise
--d "探究warmup0.25和学习率0.001是不是最优解"
