#!/bin/bash

# Function to check GPU utilization
check_gpu_utilization() {
  gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3,4,5| grep -o '[0-9]\+' | awk '{print $1}')
  echo "$gpu_utilization"
}

# Function to execute your task
execute_task() {
  # Replace this function with the logic to execute your task on the specified GPU
  echo "Executing your task on GPU"
  CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/semi/50/ODOC_semi50_align1e-1 &
  CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/semi/50/ODOC_semi50_align2e-1 &
  CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/semi/50/ODOC_semi50_align3e-1 &
  PID4=$!

  wait $PID4
  CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/semi/50/ODOC_semi50_align4e-1 &
  CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/semi/50/ODOC_semi50_align5e-1 &
  CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/semi/50/ODOC_semi50_align6e-1 &
  PID5=$!

  wait $PID5
  CUDA_VISIBLE_DEVICES=0,1 python main.py --config SEG/semi/50/ODOC_semi50_align7e-1 &
  CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEG/semi/50/ODOC_semi50_align8e-1 &
  CUDA_VISIBLE_DEVICES=4,5 python main.py --config SEG/semi/50/ODOC_semi50_align9e-1 &
  PID6=$!
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
