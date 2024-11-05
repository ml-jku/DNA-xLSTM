#!/bin/bash

# Pretrain Hyena
# See all model parameters in configs/model/hyena.yaml

cd ../ || exit 

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_DEVICES=8

# Wandb Logging
WANDB_MODE="disabled" # online, disabled
WANDB_PROJECT="xlstm_dna"

# Run script
SEQLEN=1024
MAX_STEPS=10000
D_MODEL=256
N_LAYER=4
LR="6e-4"
RC_AUG="true"

BATCH_SIZE=$(( 1048576 / SEQLEN ))
SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
TIMESTAMP=$(date +"%y%m%d%H%M%S") # use timestamp as run id
WANDB_NAME="hyena_bidir-false_rcaug-${RC_AUG}_seqlen-${SEQLEN_DIS}_dmodel-${D_MODEL}_nlayer-${N_LAYER}_lr-${LR}_runid-${TIMESTAMP}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"

mkdir -p "${HYDRA_RUN_DIR}"
python train.py \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=$(( BATCH_SIZE / NUM_DEVICES )) \
  dataset.mlm=false \
  dataset.mlm_probability=0.0 \
  dataset.rc_aug="${RC_AUG}" \
  model=hyena \
  model.d_model=${D_MODEL} \
  model.n_layer=${N_LAYER} \
  optimizer.lr="${LR}" \
  train.global_batch_size=${BATCH_SIZE} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  +trainer.val_check_interval=$(( MAX_STEPS / 10 )) \
  wandb.project="${WANDB_PROJECT}" \
  wandb.mode="${WANDB_MODE}" \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  loader.num_workers=32 \
  loader.pin_memory=false \
