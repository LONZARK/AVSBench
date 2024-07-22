
# train on avsbench s4 dataset
# --> 
cd /mnt/user/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=0 python train.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag 

# train on avsbench + (75% avsbencn + 25% synthesis_avsbench) s4 dataset
# --> 
cd /mnt/user/jiali/AVSBench/avs_scripts/avs_s4; CUDA_VISIBLE_DEVICES=0 python train_cl.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag 