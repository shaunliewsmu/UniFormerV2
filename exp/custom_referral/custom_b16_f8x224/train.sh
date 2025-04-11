#!/bin/bash

work_path=$(dirname $0)
NUM_SHARDS=1
NUM_GPUS=1  # Set to 0 to avoid distributed training issues
BATCH_SIZE=8
SAMPLING_METHOD="uniform"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${work_path}/output/test_${TIMESTAMP}_${SAMPLING_METHOD}"

export CUDA_VISIBLE_DEVICES=1

PYTHONPATH=$PYTHONPATH:./slowfast \
python3 tools/custom_test_net.py \
  --init_method tcp://localhost:10125 \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/custom_referral \
  DATA.PATH_PREFIX data/duhs-gss-split-5:v0/organized_dataset \
  DATA.SAMPLING_METHOD ${SAMPLING_METHOD} \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.CHECKPOINT_FILE_PATH $work_path/output/run_20250307_155856_uniform/best.pyth \
  TEST.ADD_SOFTMAX True \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  OUTPUT_DIR $OUTPUT_DIR