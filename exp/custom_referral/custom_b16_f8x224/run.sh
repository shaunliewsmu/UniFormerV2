#!/bin/bash

work_path=$(dirname $0)
NUM_SHARDS=1
NUM_GPUS=1
BATCH_SIZE=8

PYTHONPATH=$PYTHONPATH:./slowfast \
python3 tools/run_net.py \
  --init_method tcp://localhost:10125 \
  --cfg $work_path/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR ./data_list/custom_referral \
  DATA.PATH_PREFIX data/laryngeal_dataset_balanced/dataset \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 5 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 50 \
  SOLVER.BASE_LR 1e-5 \
  SOLVER.WARMUP_EPOCHS 5 \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.ADD_SOFTMAX True \
  OUTPUT_DIR $work_path/output