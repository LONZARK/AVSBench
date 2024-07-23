
# Test the model which trained on avsbench ms3 dataset
# conda activate avs; 
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=1 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240718-134736/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \

# Test the model which trained on avsbench + (75% avsbencn + 25% synthesis_avsbench) ms3 dataset
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=1 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240719-184238/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \


# Test the model which trained on avsbench + (50% avsbencn + 50% synthesis_avsbench) ms3 dataset
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=1 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240722-214918/checkpoints/MS3_pvt_best_original.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \
