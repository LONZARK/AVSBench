


# conda activate avs; cd /home/jiali/AVSBench/avs_scripts/avs_s4; 

# train on avsbench s4 dataset
CUDA_VISIBLE_DEVICES=3 python train.py --session_name S4_pvt --visual_backbone pvt --train_batch_size 4 --lr 0.0001 \
--tpavi_stages 0 1 2 3 --tpavi_va_flag 
# --> /home/jiali/AVSBench/avs_scripts/avs_s4/train_logs/S4_pvt_20240719-180855