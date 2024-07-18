
# setting='MS3'
# visual_backbone="pvt" # "resnet" or "pvt"

# python train.py \
#         --session_name ${setting}_${visual_backbone} \
#         --visual_backbone ${visual_backbone} \
#         --max_epoches 30 \
#         --train_batch_size 4 \
#         --lr 0.0001 \
#         --tpavi_stages 0 1 2 3 \
#         --tpavi_va_flag \
#         --masked_av_flag \
#         --masked_av_stages 0 1 2 3 \
#         --lambda_1 0.5 \
#         --kl_flag \
#         # --load_ssss_params \
#         # --trained_ssss_path "../pvt_sss/train_logs/ssss_20220218-115625/checkpoints/ssss_best.pth" \


# conda activate avs; cd /home/jiali/AVSBench/avs_scripts/avs_ms3; 

CUDA_VISIBLE_DEVICES=3 python train.py --session_name MS3_pvt --visual_backbone pvt --max_epoches 30 --train_batch_size 4 --lr 0.0001 --tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag 