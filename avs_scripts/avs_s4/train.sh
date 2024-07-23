
# # train on avsbench s4 dataset
# # --> /home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240719-191631
# cd /home/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=3 python train.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
# --tpavi_stages 0 1 2 3 --tpavi_va_flag --easy_ratio 1.0


# train on (100% avsbench s4 dataset) + (75% s4 and 25% avs_synthesis s4)
# --> /home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240722-
cd /home/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=2 python train.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --easy_ratio 0.75

# train on (100% avsbench s4 dataset) + (50% s4 and 50% avs_synthesis s4)
# --> /home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240722-
cd /home/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=2 python train.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --easy_ratio 0.5

# train on (100% avsbench s4 dataset) + (25% s4 and 75% avs_synthesis s4)
# --> /home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240722-
cd /home/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=2 python train.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --easy_ratio 0.25


# train on (100% avsbench s4 dataset) + (0% s4 and 100% avs_synthesis s4)
# --> /home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240722-
cd /home/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=2 python train.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --easy_ratio 0