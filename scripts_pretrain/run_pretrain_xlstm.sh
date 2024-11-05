#!/bin/bash

# Pretrain xLSTM
# See all model parameters in configs/model/xlstm.yaml

cd ../ || exit

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_DEVICES=8

# Wandb Logging
WANDB_MODE="disabled" # online, disabled
WANDB_PROJECT="xlstm_dna"

SEQLEN=1024
MAX_STEPS=10000
D_MODEL=256 # model embedding dim
N_LAYER=4
SLSTM_AT='[0,1,2,3]' # which of the 4 blocks are sLSTM vs. mLSTM blocks e.g. [0,1,2,3] or [0, 2]. [] disables sLSTM layers.

LR="8e-3"
BIDIR="true" # causal language modeling (CLM) vs. masked-language modeling (MLM)
BIDIR_ALTERNATING="false" # if true, alternates direction of subsequent layers. If false, use weight sharing when bidirectionality is enabled
RCPS="false" # enable RC equivariance via parameter-sharing. Set this to true for PS models. Note: currently, PS is only supported for sLSTM variants.
RC_AUG="true" # reverse complement augmentation. Set this to false for PS equivariant models.
BATCH_SIZE=(( 1048576 / SEQLEN)) # we use constant number of tokens between different context sizes

SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
TIMESTAMP=$(date +"%y%m%d%H%M%S") # use timestamp as run id
WANDB_NAME="xlstm_bidir-${BIDIR}_alt-${BIDIR_ALTERNATING}_rcaug-${RC_AUG}_RCPS-${RCPS}_seqlen-${SEQLEN_DIS}_dmodel-${D_MODEL}_nlayer-${N_LAYER}_lr-${LR}_runid-${TIMESTAMP}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"

mkdir -p "${HYDRA_RUN_DIR}"

# additional xLSTM arguments
# sLSTM arguments:
#   - s_proj_factor: GLU upprojection factor
#   - s_round_proj_up_dim_up: whether to round up to multiple. 
#   - s_round_proj_up_to_multiple_of: round to multiple in GLU upprojection.
#   - s_num_heads: number of heads in sLSTM memory mixing.
# mLSTM arguments:
#   - m_conv1d_kernel_size: kernel size of 1d convolution.
#   - m_conv1d_causal: whether the convolution is causal.
#   - m_qkv_proj_blocksize: block size in block-diagonal QKV projection matrices of mLSTM.
#   - m_num_heads: number of heads.
#   - m_proj_factor: upprojection factor in mLSTM pre-upprojection.
#   - m_backend: which backend to use, either "parallel" or "chunkwise"
#   - m_chunk_size: chunksize for chunkwise backend
#   - m_backend_bidirectional: enables native mLSTM bidirectionality.
#   - m_position_embeddings: if true, use RoPE
#   - m_bias: whether biases are enabled in linear and norm layers.

python train.py \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=$(( BATCH_SIZE / NUM_DEVICES )) \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  dataset.rc_aug="${RC_AUG}" \
  model=xlstm \
  model.config.d_model=${D_MODEL} \
  model.config.n_layer=${N_LAYER} \
  model.config.max_length=${SEQLEN} \
  model.config.s_lstm_at=${SLSTM_AT} \
  model.config.s_num_heads=4 \
  model.config.s_proj_factor=1.1 \
  model.config.s_round_proj_up_dim_up=false \
  model.config.s_round_proj_up_to_multiple_of=8 \
  model.config.m_conv1d_kernel_size=4 \
  model.config.m_conv1d_causal=true \
  model.config.m_qkv_proj_blocksize=4 \
  model.config.m_num_heads=4 \
  model.config.m_proj_factor=2.0 \
  model.config.m_backend="chunkwise" \
  model.config.m_chunk_size=1024 \
  model.config.m_backend_bidirectional=false \
  model.config.m_position_embeddings=true \
  model.config.m_bias=true \
  model.config.bidirectional=${BIDIR} \
  model.config.bidirectional_alternating=${BIDIR_ALTERNATING} \
  model.config.rcps=${RCPS} \
  optimizer.lr="${LR}" \
  optimizer.weight_decay=0.1 \
  train.global_batch_size=${BATCH_SIZE} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  trainer.gradient_clip_val=1.0 \
  trainer.precision=bf16 \
  +trainer.track_grad_norm=2 \
  +trainer.val_check_interval=$((( MAX_STEPS / 10 ))) \
  wandb.project="${WANDB_PROJECT}" \
  wandb.mode="${WANDB_MODE}" \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  loader.num_workers=32 \
  loader.pin_memory=false \