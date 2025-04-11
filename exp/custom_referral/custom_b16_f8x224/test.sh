#!/bin/bash

work_path=$(dirname $0)
NUM_SHARDS=1
NUM_GPUS=1
BATCH_SIZE=8
SAMPLING_METHOD="uniform"
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0
CHECKPOINT_PATH="${work_path}/output/checkpoints/checkpoint_best.pyth"

# Accept command line arguments to override defaults
while [[ $# -gt 0 ]]; do
  case $1 in
    --alpha)
      FOCAL_ALPHA="$2"
      shift 2
      ;;
    --gamma)
      FOCAL_GAMMA="$2"
      shift 2
      ;;
    --sampling_method)
      SAMPLING_METHOD="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${work_path}/output/test_${TIMESTAMP}_${SAMPLING_METHOD}_alpha${FOCAL_ALPHA}_gamma${FOCAL_GAMMA}"

PYTHONPATH=$PYTHONPATH:./slowfast \
python3 tools/custom_test_net.py \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/custom_referral \
  DATA.PATH_PREFIX data/duhs-gss-split-5:v0/organized_dataset \
  DATA.SAMPLING_METHOD ${SAMPLING_METHOD} \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.CHECKPOINT_FILE_PATH ${CHECKPOINT_PATH} \
  TEST.ADD_SOFTMAX True \
  TEST.METRICS.USE_BALANCED_METRICS True \
  MODEL.LOSS_FUNC "focal_loss" \
  MODEL.FOCAL_LOSS.ENABLE True \
  MODEL.FOCAL_LOSS.ALPHA ${FOCAL_ALPHA} \
  MODEL.FOCAL_LOSS.GAMMA ${FOCAL_GAMMA} \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  OUTPUT_DIR $OUTPUT_DIR