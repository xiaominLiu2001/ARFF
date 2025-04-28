#!/bin/sh

for STAGE in 1 2; do
  python ./train_2_cls.py \
    --dataset AD \
    --num_classes 4 \
    --fold_id 0 \
    --data_csv /home/liuxm/Iron_Reg_Model/Utils/AD_LUMC_csv/cluster4_AD_LUMC_csv \
    --data_split_json /home/liuxm/Iron_Reg_Model/Utils/87_split_results.json \
    --feat_size 1024 \
    --train_stage ${STAGE} \
    --checkpoint_pretrained /home/liuxm/Iron_Reg_Model/Reg_model/execute/985_87_fold0_clus4_results_AD/AD_87/train_1/CLAM_SB/stage_3/model_best.pth.tar \
    --T 6 \
    --scheduler CosineAnnealingLR \
    --optimizer Adam \
    --picked_method acc \
    --batch_size 1 \
    --epochs 200 \
    --backbone_lr 0.00005 \
    --fc_lr 0.00002 \
    --num_clusters 4 \
    --device 3 \
    --seed 985 \
    --base_save_dir './985_87_fold0_clus4_results_AD'
done
python ./train_2_cls.py \
  --dataset AD_87 \
  --fold_id 3 \
  --data_csv /home/liuxm/Iron_Reg_Model/Utils/AD_LUMC_csv/cluster4_AD_LUMC_csv \
  --data_split_json /home/liuxm/Iron_Reg_Model/Utils/87_split_results.json \
  --feat_size 1024 \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --optimizer Adam \
  --picked_method acc \
  --batch_size 1 \
  --epochs 100 \
  --backbone_lr 0.0001 \
  --fc_lr 0.00005 \
  --device 3 \
  --num_clusters 4 \
  --seed 985 \
  --base_save_dir './985_87_fold3_clus4_results_AD'
