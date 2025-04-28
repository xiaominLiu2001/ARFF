#!/bin/sh

for STAGE in 1 2; do
  python ./train_1.py \
    --dataset AD_LUMC \
    --fold_id 0 \
    --base_save_dir ./fold0_cluster6_results_AD \
    --data_csv /your_path_eg_cluster6_AD_LUMC_csv \
    --data_split_json /your_path_eg_AD_LUMC_fold0_split.json \
    --train_stage ${STAGE} \
    --backbone_lr 0.0002 \
    --fc_lr 0.0001 \
    --device 0 \
    --num_clusters 6 \
    --seed 2021
done

python ./train_1.py \
  --dataset AD_LUMC \
  --fold_id 0 \
  --base_save_dir ./fold0_cluster6_results_AD \
  --data_csv /your_path_eg_cluster6_AD_LUMC_csv \
  --data_split_json /your_path_eg_AD_LUMC_fold0_split.json \
  --train_stage 3 \
  --T 6 \
  --backbone_lr 0.0005 \
  --fc_lr 0.00003 \
  --device 0 \
  --num_clusters 6 \
  --seed 2021
