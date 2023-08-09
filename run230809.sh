# done
#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_semi512x512/Teacher_Student/random1_ODOC_semi90_None &
#CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config SEG/cropped_semi512x512/Teacher_Student/random1_ODOC_semi90_None_weightedLoss

#CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_sup512x512/random1_ODOC_sup10_DCBDLoss



CUDA_VISIBLE_DEVICES=0 python main.py -ng 1 --config SEG/cropped_semi512x512/Teacher_Student/random1_ODOC_semi90_PretrainT_PretrainS_None &
CUDA_VISIBLE_DEVICES=1 python main.py -ng 1 --config SEG/cropped_semi512x512/Teacher_Student/random1_ODOC_semi90_scratchT_scratchS_None
