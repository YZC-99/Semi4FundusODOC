#!/bin/bash

# Function to check GPU utilization
check_gpu_utilization() {
  gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 2,3| grep -o '[0-9]\+' | awk '{print $1}')
  echo "$gpu_utilization"
}

# Function to execute your task
execute_task() {
  # Replace this function with the logic to execute your task on the specified GPU
  echo "Executing your task on GPU"
  CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_tgt_prototype
  CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/cropped_semi512x512/res50deeplabv3plus/random1_ODOC_semi90_B_CJ_CO_RG_tgt_prototype
}

# Main loop
while true; do
  # Check GPU utilization
  utilization=$(check_gpu_utilization)

  # Check if all GPUs are idle
  all_idle=true
  for util in $utilization; do
    if [ "$util" -ne 0 ]; then
      all_idle=false
      echo "all idle"
      break
    fi
  done

  if $all_idle; then
    # check again
    utilization=$(check_gpu_utilization)
    # Check if all GPUs are idle
    all_idle=true
    for util in $utilization; do
      if [ "$util" -ne 0 ]; then
        all_idle=false
        echo "all idle"
        break
      fi
    done
    # Execute your task here
    execute_task
    break
  fi

  # Wait for some time before checking again
  sleep 100  # You can adjust the interval as needed
done
