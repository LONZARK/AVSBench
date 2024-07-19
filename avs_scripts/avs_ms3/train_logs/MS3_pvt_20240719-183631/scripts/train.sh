


# conda activate avs; cd /home/jiali/AVSBench/avs_scripts/avs_ms3; 

# train on avsbench ms3 dataset
CUDA_VISIBLE_DEVICES=2 python train.py --session_name MS3_pvt --visual_backbone pvt --max_epoches 30 --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag 
# --> /home/jiali/AVSBench/avs_scripts/avs_ms3/train_logs/MS3_pvt_20240718-134736