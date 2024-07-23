
# train on avsbench ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240718-134736
# cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python train.py --session_name MS3_pvt --visual_backbone pvt --max_epoches 30 --train_batch_size 4 --lr 0.0001 \
# --tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag --easy_ratio 1.0


# train on avsbench + (75% avsbencn + 25% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240722-232303
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python train.py --session_name MS3_pvt --visual_backbone pvt --max_epoches 30 --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag --easy_ratio 0.75

# train on avsbench + (50% avsbencn + 50% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240722-234906
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python train.py --session_name MS3_pvt --visual_backbone pvt --max_epoches 30 --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag --easy_ratio 0.5

# train on avsbench + (25% avsbencn + 75% synthesis_avsbench) ms3 dataset
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240723-001517
cd /home/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=3 python train.py --session_name MS3_pvt --visual_backbone pvt --max_epoches 30 --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag --easy_ratio 0.25