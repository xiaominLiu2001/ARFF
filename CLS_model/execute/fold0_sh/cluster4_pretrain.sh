#!/bin/sh

for STAGE in 1 2; do
  python ./train_1.py \
    --dataset AD \
    --fold_id 3 \
    --base_save_dir ./985_fold3_clus4_results_AD \
    --data_csv /home/liuxm/Iron_Reg_Model/Utils/AD_LUMC_csv/cluster4_AD_LUMC_csv \
    --data_split_json /home/liuxm/Iron_Reg_Model/Utils/87_split_results.json \
    --feat_size 1024 \
    --train_stage ${STAGE} \
    --T 6 \
    --scheduler CosineAnnealingLR \
    --optimizer Adam \
    --batch_size 128 \
    --epochs 300 \
    --backbone_lr 0.0002 \
    --fc_lr 0.0001 \
    --device 2,3 \
    --patience 100 \
    --num_clusters 4 \
    --seed 985
done

python ./train_1.py \
  --dataset AD \
  --fold_id 3 \
  --base_save_dir ./985_fold3_clus4_results_AD \
  --data_csv /home/liuxm/Iron_Reg_Model/Utils/AD_LUMC_csv/cluster4_AD_LUMC_csv \
  --data_split_json /home/liuxm/Iron_Reg_Model/Utils/87_split_results.json \
  --feat_size 1024 \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --optimizer Adam \
  --batch_size 128 \
  --epochs 300 \
  --backbone_lr 0.0005 \
  --fc_lr 0.00003 \
  --device 2,3 \
  --patience 100 \
  --num_clusters 4 \
  --seed 985
