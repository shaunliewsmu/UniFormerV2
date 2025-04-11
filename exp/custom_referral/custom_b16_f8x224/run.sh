#!/bin/bash

work_path=$(dirname $0)
NUM_SHARDS=1
NUM_GPUS=2  # Set to 0 for non-distributed training
BATCH_SIZE=8
SAMPLING_METHOD="uniform"
FOCAL_ALPHA=0.25  # Default focal loss alpha (can be overridden)
FOCAL_GAMMA=2.0   # Default focal loss gamma (can be overridden)

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
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${work_path}/output/run_${TIMESTAMP}_${SAMPLING_METHOD}_alpha${FOCAL_ALPHA}_gamma${FOCAL_GAMMA}"

echo "Running with focal loss parameters: alpha=${FOCAL_ALPHA}, gamma=${FOCAL_GAMMA}"
echo "Output directory: ${OUTPUT_DIR}"

PYTHONPATH=$PYTHONPATH:./slowfast \
python3 tools/run_net.py \
  --cfg $work_path/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR ./data_list/custom_referral \
  DATA.PATH_PREFIX data/duhs-gss-split-5:v0/organized_dataset \
  DATA.SAMPLING_METHOD ${SAMPLING_METHOD} \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 5 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 50 \
  SOLVER.BASE_LR 1e-5 \
  SOLVER.WARMUP_EPOCHS 1 \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.ADD_SOFTMAX True \
  MODEL.LOSS_FUNC "focal_loss" \
  MODEL.FOCAL_LOSS.ENABLE True \
  MODEL.FOCAL_LOSS.ALPHA ${FOCAL_ALPHA} \
  MODEL.FOCAL_LOSS.GAMMA ${FOCAL_GAMMA} \
  OUTPUT_DIR $OUTPUT_DIR