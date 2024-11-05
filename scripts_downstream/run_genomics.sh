# Genomic Benchmark

cd ../ || exit

# environment setup
#export PATH=/system/apps/userenv/schimune/caduceus_env2/bin:${PATH}
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# Example configurations:

## Hyena
## TODO: Download HF model from https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen
#CONFIG_PATH=".../model_config.json"
#PRETRAINED_PATH=".../checkpoints/last.ckpt"
#DISPLAY_NAME="hyena"
#MODEL="hyena"
#MODEL_NAME="dna_embedding"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "false" "true" )
#LRS=( "6e-4" )

## Mamba CLM
#CONFIG_PATH=".../model_config.json"
#PRETRAINED_PATH=".../checkpoints/last.ckpt"
#DISPLAY_NAME="mamba_uni"
#MODEL="mamba"
#MODEL_NAME="dna_embedding_mamba"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "1e-3" "2e-3" )

## Caduceus no PH
#CONFIG_PATH=".../model_config.json"
#PRETRAINED_PATH=".../checkpoints/last.ckpt"
#DISPLAY_NAME="caduceus_NO_PH"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "2e-3")

## Caduceus PH
#CONFIG_PATH=".../model_config.json"
#PRETRAINED_PATH=".../checkpoints/last.ckpt"
#DISPLAY_NAME="caduceus_ph"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="true"
#RC_AUGS=( "false" )

## Caduceus PS
#CONFIG_PATH=".../model_config.json"
#PRETRAINED_PATH=".../checkpoints/last.ckpt"
#DISPLAY_NAME="caduceus_ps"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
#CONJOIN_TEST="false"
#RC_AUGS=( "false" )

# Wandb Logging
WANDB_MODE="disabled" # online, disabled

MODEL="xlstm" # "caduceus", "mamba", "hyena"
MODEL_NAME="dna_embedding_xlstm" # "dna_embedding_mamba", "dna_embedding_caduceus", "dna_embedding" (hyena)
TIMESTAMP=$(date +"%y_%m%_d%_H%M%S") # use timestamp as run id

# PH models
WANDB_PROJECT="genomic_benchmark_PH"
CONFIG_PATH=$(realpath "./checkpoints/context_1k/dna_xlstm_500k_mlm_ph/model_config.json")
PRETRAINED_PATH=$(realpath "./checkpoints/context_1k/dna_xlstm_500k_mlm_ph/checkpoints/test/loss.ckpt")
DISPLAY_NAME="xlstm_ph"
CONJOIN_TRAIN_DECODER="false"  # Use this in decoder to always combine forward and reverse complement channels
CONJOIN_TEST="true"
RC_AUGS=( "true" "false" )
LRS=("1e-3" "2e-3" "4e-4" "6e-4" "8e-4" ) # define learning rates for sweep

# PS models (xLSTM and Caduceus only)
#WANDB_PROJECT="genomic_benchmark_PS"
#CONFIG_PATH=$(realpath "./model_checkpoints/context_1k/dna_xlstm_500k_mlm_ps/model_config.json")
#PRETRAINED_PATH=$(realpath "./model_checkpoints/context_1k/dna_xlstm_500k_mlm_ps/checkpoints/test/loss.ckpt")
#DISPLAY_NAME="xlstm_ps"
#CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
#CONJOIN_TEST="false"
#RC_AUGS=( "false" )
#LRS=("1e-3" "2e-3" "4e-4" "6e-4" "8e-4" ) # define learning rates for sweep

## Tasks:
# "dummy_mouse_enhancers_ensembl"
# "demo_coding_vs_intergenomic_seqs"
# "demo_human_or_worm"
# "human_enhancers_cohn"
# "human_enhancers_ensembl"
# "human_ensembl_regulatory"
# "human_ocr_ensembl"
# "human_nontata_promoters"

for TASK in "dummy_mouse_enhancers_ensembl"; do # select task or list of tasks from task list above
  for LR in "${LRS[@]}"; do
    for BATCH_SIZE in 64 128; do
      for RC_AUG in "${RC_AUGS[@]}"; do
        WANDB_NAME="${DISPLAY_NAME}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-${RC_AUG}_runid-${TIMESTAMP}"
        for seed in $(seq 1 5); do
          HYDRA_RUN_DIR="./outputs/downstream/gb_cv5/${DISPLAY_NAME}/${TASK}/${WANDB_NAME}/seed-${seed}"
          mkdir -p "${HYDRA_RUN_DIR}"
          echo "*****************************************************"
          echo "Running GenomicsBenchmark model: ${DISPLAY_NAME}, task: ${TASK}, lr: ${LR}, batch_size: ${BATCH_SIZE}, rc_aug: ${RC_AUG}, SEED: ${seed}"
          python train.py \
            experiment=hg38/genomic_benchmark \
            callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
            dataset.dataset_name="${TASK}" \
            dataset.train_val_split_seed=${seed} \
            dataset.batch_size=${BATCH_SIZE} \
            dataset.rc_aug="${RC_AUG}" \
            +dataset.conjoin_train=false \
            +dataset.conjoin_test="${CONJOIN_TEST}" \
            model="${MODEL}" \
            model._name_="${MODEL_NAME}" \
            +model.config_path="${CONFIG_PATH}" \
            +model.conjoin_test="${CONJOIN_TEST}" \
            +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
            +decoder.conjoin_test="${CONJOIN_TEST}" \
            optimizer.lr="${LR}" \
            trainer.max_epochs=10 \
            train.pretrained_model_path="${PRETRAINED_PATH}" \
            train.pretrained_model_strict_load=false \
            wandb.group="downstream/gb_cv5" \
            wandb.job_type="${TASK}" \
            wandb.name="${WANDB_NAME}" \
            wandb.id="gb_cv5_${TASK}_${WANDB_NAME}_seed-${seed}" \
            +wandb.tags=\["seed-${seed}"\] \
            hydra.run.dir="${HYDRA_RUN_DIR}" \
            wandb.project="${WANDB_PROJECT}" \
            wandb.mode="${WANDB_MODE}" \
            loader.num_workers=16
          echo "*****************************************************"
        done
        #----
      done
    done
  done
done