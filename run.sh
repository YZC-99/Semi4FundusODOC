#!/bin/bash

# 设置需要的总循环次数
total_loops=150  # 例如，这里设置为10，您可以根据需要进行更改

# 设置并行训练的显卡数量
num_gpus=3

# 循环从0开始，步长为3，作为随机种子seed
for seed in $(seq 0 3 $((total_loops-1))); do
  # 并行训练三个不同随机种子的模型
  for gpu_id in $(seq 0 $((num_gpus-1))); do
    # 启动训练任务，这里假设您有一个train.py脚本用于模型训练
    # 您需要根据实际情况修改训练命令和参数
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py -s $((seed+gpu_id)) --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.4 --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/noise &
    sleep 15
  done

  # 等待上一轮的所有训练任务完成
  wait
done

echo "所有训练任务已完成"


total_loops=12  # 例如，这里设置为12，每次增加0.1会循环10次

# 设置并行训练的显卡数量
num_gpus=3

# 循环从0开始，步长为0.1（转换为浮点数），作为随机种子seed
for ((i=5; i<=total_loops; i+=3)); do
  for gpu_id in $(seq 0 $((num_gpus-1))); do
    FC_loss_value=$(awk "BEGIN{print ($i + $gpu_id) * 0.1}")
    # 启动训练任务，这里假设您有一个train.py脚本用于模型训练
    # 您需要根据实际情况修改训练命令和参数
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --lr 4.0e-4 --warmup 1e-2 --BD_loss 2.0 --FC_loss $FC_loss_value --epochs 100 --scheduler poly --config Drishti-GS/cropped_sup256x256/my_segformer/v7-ii-1-6-v1/Base &
    sleep 15
  done

  # 等待上一轮的所有训练任务完成
  wait
done

echo "所有训练任务已完成"
