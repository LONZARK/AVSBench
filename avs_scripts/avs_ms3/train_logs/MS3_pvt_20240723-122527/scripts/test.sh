
# Test the model which trained on avsbench ms3 dataset
# conda activate avs; 
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240718-134736
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240718-134736/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \

# Test the model which trained on avsbench + (75% avsbencn + 25% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-092853
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240722-232303/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \

# Test the model which trained on avsbench + (50% avsbencn + 50% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-092915
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240722-234906/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \

# Test the model which trained on avsbench + (25% avsbencn + 75% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-092936
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-001517/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \

# Test the model which trained on avsbench + (75% avsbencn + 25% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-093134
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240722-232303/checkpoints/MS3_pvt_final.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \

# Test the model which trained on avsbench + (50% avsbencn + 50% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-093158
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240722-234906/checkpoints/MS3_pvt_final.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \

# Test the model which trained on avsbench + (25% avsbencn + 75% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-093216
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-001517/checkpoints/MS3_pvt_final.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask \




# ===================================================
# Test the model which trained on avsbench ms3 dataset
# ------ dataset ms3 original dataset
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240718-134736
# ------ dataset synthesis ms3 dataset
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-094046
# ------ dataset ms3 frames + slient audio
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-102057


# ===================================================
# Test the model which trained on avsbench + (75% avsbencn + 25% synthesis_avsbench) ms3 dataset
# ------ dataset ms3 original dataset
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-092853
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-093134
# ------ dataset synthesis ms3 dataset
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-094109
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-094209
# ------ dataset ms3 frames + slient audio
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-102115
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-102219

# ===================================================
# Test the model which trained on avsbench + (50% avsbencn + 50% synthesis_avsbench) ms3 dataset
# ------ dataset ms3 original dataset
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-092915
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-093158
# ------ dataset synthesis ms3 dataset
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-094128
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-094232
# ------ dataset ms3 frames + slient audio
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-102137
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-102239

# ===================================================
# Test the model which trained on avsbench + (25% avsbencn + 75% synthesis_avsbench) ms3 dataset
# ------ dataset ms3 original dataset
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-092936
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-093216
# ------ dataset synthesis ms3 dataset
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-094151
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-094250
# ------ dataset ms3 frames + slient audio
# Best --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-102157
# Final --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-102300