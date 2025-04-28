#!/bin/sh

for STAGE in 1 2; do
  python ./train_2_cls.py \
    --dataset AD_LUMC \
    --num_classes 4 \
    --fold_id 0 \
    --data_csv /your_path_eg_cluster6_AD_LUMC_csv \
    --data_split_json /your_path_eg_AD_LUMC_fold0_split.json \
    --train_stage ${STAGE} \
    --checkpoint_pretrained /your_path/ARFF/CLS_model/execute/fold0_cluster6_results_AD/AD_LUMC/train_1/CLAM_SB/stage_3/model_best.pth.tar \
    --picked_method acc \
    --backbone_lr 0.00005 \
    --fc_lr 0.00002 \
    --num_clusters 6 \
    --device 0 \
    --seed 2021 \
    --base_save_dir './fold0_cluster6_results_AD'
done
python ./train_2_cls.py \
  --dataset AD_LUMC \
  --fold_id 0 \
  --data_csv /your_path_eg_cluster6_AD_LUMC_csv \
  --data_split_json /your_path_eg_AD_LUMC_fold0_split.json \
  --train_stage 3 \
  --picked_method acc \
  --backbone_lr 0.0001 \
  --fc_lr 0.00005 \
  --device 0 \
  --num_clusters 6 \
  --seed 2021 \
  --base_save_dir './fold0_cluster6_results_AD'
