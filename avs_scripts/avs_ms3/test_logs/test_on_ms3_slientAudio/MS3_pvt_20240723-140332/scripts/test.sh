
# Test the model which trained on avsbench ms3 dataset
# conda activate avs; 
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240718-134736
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240718-134736/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 

# Test the model which trained on avsbench + (75% avsbencn + 25% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-110640
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-110640/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-110640/checkpoints/MS3_pvt_final.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 


# Test the model which trained on avsbench + (50% avsbencn + 50% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-113246
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-113246/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-113246/checkpoints/MS3_pvt_final.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 


# Test the model which trained on avsbench + (25% avsbencn + 75% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-115908
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-115908/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-115908/checkpoints/MS3_pvt_final.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 



# Test the model which trained on avsbench + (0% avsbencn + 100% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/test_logs/MS3_pvt_20240723-122527
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-122527/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python test.py --session_name MS3_pvt --visual_backbone pvt \
    --weights "/home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-122527/checkpoints/MS3_pvt_final.pth" \
    --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 




# ===================================================
# Test the model which trained on avsbench ms3 dataset
# ------ dataset ms3 original dataset
# ------ dataset synthesis ms3 dataset
# ------ dataset ms3 frames + slient audio


# ===================================================
# Test the model which trained on avsbench + (75% avsbencn + 25% synthesis_avsbench) ms3 dataset
# ------ dataset ms3 original dataset
# ------ dataset synthesis ms3 dataset
# ------ dataset ms3 frames + slient audio

# ===================================================
# Test the model which trained on avsbench + (50% avsbencn + 50% synthesis_avsbench) ms3 dataset
# ------ dataset ms3 original dataset
# ------ dataset synthesis ms3 dataset
# ------ dataset ms3 frames + slient audio

# ===================================================
# Test the model which trained on avsbench + (25% avsbencn + 75% synthesis_avsbench) ms3 dataset
# ------ dataset ms3 original dataset
# ------ dataset synthesis ms3 dataset
# ------ dataset ms3 frames + slient audio