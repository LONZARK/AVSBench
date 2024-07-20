
# train on avsbench ms3 dataset
# --> 
cd /mnt/user/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=0 python train.py --session_name MS3_pvt --visual_backbone pvt --max_epoches 30 --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag 


# train on avsbench + (75% avsbencn + 25% synthesis_avsbench) ms3 dataset
# --> 
cd /mnt/user/jiali/AVSBench/avs_scripts/avs_ms3; CUDA_VISIBLE_DEVICES=0 python train_cl.py --session_name MS3_pvt --visual_backbone pvt --max_epoches 30 --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag 


# train on avsbench s4 dataset
# --> 
cd /mnt/user/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=0 python train.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag 

# train on avsbench + (75% avsbencn + 25% synthesis_avsbench) s4 dataset
# --> 
cd /mnt/user/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=0 python train_cl.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag 