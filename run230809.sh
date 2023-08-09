# done
CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_semi512x512/Teacher_Student/random1_ODOC_semi90_None &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config SEG/cropped_semi512x512/Teacher_Student/random1_ODOC_semi90_None_weightedLoss
