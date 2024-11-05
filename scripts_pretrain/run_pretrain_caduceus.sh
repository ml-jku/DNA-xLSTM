#!/bin/bash

# Pretrain Caduceus
# See all model parameters in configs/model/caduceus.yaml

cd ../ || exit

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_DEVICES=8

# Wandb Logging
WANDB_MODE="disabled" # online, disabled
WANDB_PROJECT="xlstm_dna"

SEQLEN=1024
MAX_STEPS=10000
D_MODEL=256
N_LAYER=4
LR="8e-3"
BIDIRECTIONAL="true" # CLM or MLM
BIDIRECTIONAL_STRATEGY="add"
BIDIRECTIONAL_WEIGHT_TIE="true"
RCPS="true" # enable RC equivariance via parameter-sharing. Set this to true for PS models.
RC_AUG="false" # reverse complement augmentation. Set this to false for PS equivariant models.
BATCH_SIZE=$(( 1048576 / SEQLEN )) # we use constant number of tokens between different context sizes

SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
TIMESTAMP=$(date +"%y%m%d%H%M%S") # use timestamp as run id
WANDB_NAME="caduceus_bidir-${BIDIR}_rcaug-${RC_AUG}_RCPS-${RCPS}_seqlen-${SEQLEN_DIS}_dmodel-${D_MODEL}_nlayer-${N_LAYER}_lr-${LR}_runid-${TIMESTAMP}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"
mkdir -p "${HYDRA_RUN_DIR}"

python -m train \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=$(( BATCH_SIZE / 8 )) \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  dataset.rc_aug="${RC_AUG}" \
  model="caduceus" \
  model.config.d_model=${D_MODEL} \
  model.config.n_layer=${N_LAYER} \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=${BIDIRECTIONAL_STRATEGY} \
  model.config.bidirectional_weight_tie=${BIDIRECTIONAL_WEIGHT_TIE} \
  model.config.rcps=${RCPS} \
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
