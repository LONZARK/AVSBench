

# train on avsbench s4 dataset
cd /home/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=2 python test.py --session_name S4_pvt --visual_backbone pvt \
--weights "/home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240719-191631/S4_pvt_best.pth" --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 

# train on (100% avsbench s4 dataset) + (75% s4 and 25% avs_synthesis s4)
cd /home/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=2 python test.py --session_name S4_pvt --visual_backbone pvt \
--weights "/home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240722-211826/S4_pvt_best.pth" --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 

# train on avsbench s4 dataset
cd /home/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=2 python test.py --session_name S4_pvt --visual_backbone pvt \
--weights "/home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240722-211952/S4_pvt_best.pth" --test_batch_size 4 --tpavi_stages 0 1 2 3 --tpavi_va_flag --save_pred_mask 
