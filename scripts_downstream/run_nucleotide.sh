# Nucleotide Transformer Benchmark

cd ../ || exit

# environment setup
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# Example configurations:

## Caduceus NO POST HOC
#CONFIG_PATH=".../model_config.json"
#PRETRAINED_PATH=".../checkpoints/last.ckpt"
#DISPLAY_NAME="caduceus_NO_PH"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "1e-3" "2e-3")

## Caduceus Post-Hoc
#CONFIG_PATH=".../model_config.json"
#PRETRAINED_PATH=".../checkpoints/last.ckpt"
#DISPLAY_NAME="caduceus_ph"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="true"
#RC_AUGS=( "false" )
#LRS=( "1e-3" "2e-3" )

## Caduceus Parameter Sharing
#CONFIG_PATH=".../model_config.json"
#PRETRAINED_PATH=".../checkpoints/last.ckpt"
#DISPLAY_NAME="caduceus_ps"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
#CONJOIN_TEST="false"
#RC_AUGS=( "false" )
#LRS=( "1e-3" "2e-3" )

# Wandb Logging
WANDB_MODE="disabled" # online, disabled

MODEL="xlstm"
MODEL_NAME="dna_embedding_xlstm"
TIMESTAMP=$(date +"%y_%m_%d_%H%M%S") # use timestamp as runid

# PH models
WANDB_PROJECT="nucleotides_benchmark_PH"
CONFIG_PATH=$(realpath "./checkpoints/context_1k/dna_xlstm_2m_mlm_ph/model_config.json")
PRETRAINED_PATH=$(realpath "checkpoints/context_1k/dna_xlstm_2m_mlm_ph/checkpoints/test/loss.ckpt")
DISPLAY_NAME="xlstm_ph"
CONJOIN_TRAIN_DECODER="false"  # Use this in decoder to always combine forward and reverse complement channels
CONJOIN_TEST="true"
RC_AUGS=( "false" ) 
LRS=( "1e-3" "2e-3" "4e-4" "6e-4" "8e-4")

# PS models (xLSTM and Caduceus only)
#WANDB_PROJECT="nucleotides_benchmark_PS"
#CONFIG_PATH=$(realpath "./model_checkpoints/context_1k/dna_xlstm_2m_mlm_ps/model_config.json")
#PRETRAINED_PATH=$(realpath "model_checkpoints/context_1k/dna_xlstm_2m_mlm_ps/checkpoints/test/loss.ckpt")
#DISPLAY_NAME="xlstm_ps"
#CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
#CONJOIN_TEST="false"
#RC_AUGS=( "false" )
#LRS=( "1e-3" "2e-3" "4e-4" "6e-4" "8e-4")

## Tasks:
# "H3"
# "H3K14ac"
# "H3K36me3"
# "H3K4me1"
# "H3K4me2"
# "H3K4me3"
# "H3K79me3"
# "H3K9ac"
# "H4ac"
# "enhancers"
# "enhancers_types"
# "promoter_all"
# "promoter_no_tata"
# "promoter_tata"
# "splice_sites_all"
# "splice_sites_acceptors"
# "splice_sites_donors"

for TASK in "H3"; do # select task or list of tasks from task list above
  for LR in "${LRS[@]}"; do
    for BATCH_SIZE in 64 128; do
      for RC_AUG in "${RC_AUGS[@]}"; do
        #------
        WANDB_NAME="${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}_RUNID-${TIMESTAMP}"
        for seed in $(seq 1 10); do
          HYDRA_RUN_DIR="./outputs/downstream/nt_cv10_ep20/${TASK}/${DISPLAY_NAME}/${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}_RUNID-${TIMESTAMP}/seed-${seed}"
          mkdir -p "${HYDRA_RUN_DIR}"
          echo "*****************************************************"
          echo "Running NT model: ${DISPLAY_NAME}, TASK: ${TASK}, LR: ${LR}, BATCH_SIZE: ${BATCH_SIZE}, RC_AUG: ${RC_AUG}, SEED: ${seed}"
          python train.py \
            experiment=hg38/nucleotide_transformer \
            callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
            dataset.dataset_name="${TASK}" \
            dataset.train_val_split_seed=${seed} \
            dataset.batch_size=$((BATCH_SIZE / 2)) \
            dataset.rc_aug="${RC_AUG}" \
            +dataset.conjoin_test="${CONJOIN_TEST}" \
            model="${MODEL}" \
            model._name_="${MODEL_NAME}" \
            +model.config_path="${CONFIG_PATH}" \
            +model.conjoin_test="${CONJOIN_TEST}" \
            +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
            +decoder.conjoin_test="${CONJOIN_TEST}" \
            optimizer.lr="${LR}" \
            train.pretrained_model_path="${PRETRAINED_PATH}" \
            train.pretrained_model_strict_load=false \
            train.global_batch_size=${BATCH_SIZE} \
            trainer.max_epochs=20 \
            wandb.group="downstream/nt_cv10_ep20" \
            wandb.job_type="${TASK}" \
            wandb.name="${WANDB_NAME}" \
            wandb.id="nt_cv10_ep-20_${TASK}_${WANDB_NAME}_seed-${seed}" \
            +wandb.tags=\["seed-${seed}"\] \
            hydra.run.dir="${HYDRA_RUN_DIR}" \
            wandb.project="${WANDB_PROJECT}" \
            wandb.mode="${WANDB_MODE}" \
            loader.num_workers=8
          echo "*****************************************************"
        done
        #------
      done
    done
  done
done