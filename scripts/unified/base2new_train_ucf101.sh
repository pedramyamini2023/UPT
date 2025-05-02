#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=Unified

DATASET=ucf101

CFG=vit_b16_ep200
SHOTS=16

for SEED in 3
do
    DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    CUDA_VISIBLE_DEVICES=7 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
done