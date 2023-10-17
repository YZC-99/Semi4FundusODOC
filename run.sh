CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 1 --Attention dec_transpose --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "1-dec_transpose" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 2 --Attention dec_transpose --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "2-dec_transpose" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 3 --Attention dec_transpose --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "3-dec_transpose" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 4 -s 0 --Attention dec_transpose --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "4-dec_transpose" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 5 --Attention dec_transpose --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "5-dec_transpose" &




CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-5 --backbone b4 --early_stop -1 --sample 1 --Attention org --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "1-org" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-5 --backbone b4 --early_stop -1 --sample 2 --Attention org --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "2-org" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-5 --backbone b4 --early_stop -1 --sample 3 --Attention org --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "3-org" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-5 --backbone b4 --early_stop -1 --sample 4 --Attention org --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "4-org" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-5 --backbone b4 --early_stop -1 --sample 5 --Attention org --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "5-org" &
sleep 15






CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 -s 3407 --early_stop -1 --sample 1 --Attention dec_transpose_FAMIFM_CBAM_CCA --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "1-dec_transpose_FAMIFM_CBAM_CCA" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 2 --Attention dec_transpose_FAMIFM_CBAM_CCA --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "2-dec_transpose_FAMIFM_CBAM_CCA" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 3 --Attention dec_transpose_FAMIFM_CBAM_CCA --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "3-dec_transpose_FAMIFM_CBAM_CCA" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 4 --Attention dec_transpose_FAMIFM_CBAM_CCA --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "4-dec_transpose_FAMIFM_CBAM_CCA" &
sleep 15
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1.0e-4 --backbone b4 --early_stop -1 --sample 5 --Attention dec_transpose_FAMIFM_CBAM_CCA --epochs 100 --warmup 0.01  --CE_loss 1.0 --BD_loss 0.0 --DC_loss 0.0 --FC_loss 0.0 --ContrastCrossPixelCorrect_loss 0.0 --CEpair_loss 0.0 --scheduler poly --config REFUGE/cropped_sup256x256/400/noise --d "5-dec_transpose_FAMIFM_CBAM_CCA" &
sleep 15