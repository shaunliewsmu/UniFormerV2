#!/bin/bash

work_path=$(dirname $0)
NUM_SHARDS=1
NUM_GPUS=2  # Set to 0 for non-distributed training
BATCH_SIZE=8
SAMPLING_METHOD="random"

# Data augmentation parameters
AUGMENTATION_ENABLE=true
AUGMENTATION_METHOD="random"
# MAX_AUG_ROUNDS=5  # Set to your preferred number or leave unset for auto-calculation

FOCAL_ALPHA=1.0  # Default focal loss alpha (can be overridden)
FOCAL_GAMMA=4.0   # Default focal loss gamma (can be overridden)
# above alpha and gamma for bagls dataset
# FOCAL_ALPHA=0.25  # Default focal loss alpha
# FOCAL_GAMMA=2.0   # Default focal loss gamma 
#above alpha and gamma for duke dataset

# DATALIST_PATH="./data_list/bagls-split"  # Path to the data list
# DATASET_PATH="data/bagls-split:v0/dataset"  # Path to the dataset
DATALIST_PATH="./data_list/balanced-duke-using-duke"  # Path to the data list
DATASET_PATH="data/balanced-dataset"  # Path to the dataset
# DATALIST_PATH="./data_list/custom_referral"  # Path to the data list
# DATASET_PATH="data/duhs-gss-split-5:v0/organized_dataset"  # Path to the dataset

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
    --augmentation)
      AUGMENTATION_ENABLE=true
      shift
      ;;
    --augmentation_method)
      AUGMENTATION_METHOD="$2"
      shift 2
      ;;
    --max_aug_rounds)
      MAX_AUG_ROUNDS="$2"
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

# Add augmentation info to output directory name if enabled
if [ "$AUGMENTATION_ENABLE" = true ]; then
  OUTPUT_DIR="${OUTPUT_DIR}_aug_${AUGMENTATION_METHOD}"
  if [ -n "$MAX_AUG_ROUNDS" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_r${MAX_AUG_ROUNDS}"
  fi
fi

echo "Running with focal loss parameters: alpha=${FOCAL_ALPHA}, gamma=${FOCAL_GAMMA}"
echo "Output directory: ${OUTPUT_DIR}"

if [ "$AUGMENTATION_ENABLE" = true ]; then
  echo "Data augmentation enabled with method: ${AUGMENTATION_METHOD}"
  if [ -n "$MAX_AUG_ROUNDS" ]; then
    echo "Maximum augmentation rounds: ${MAX_AUG_ROUNDS}"
  else
    echo "Maximum augmentation rounds: auto-calculated"
  fi
fi

# Build command
CMD="PYTHONPATH=$PYTHONPATH:./slowfast \
  python3 tools/run_net.py \
  --cfg $work_path/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR ${DATALIST_PATH} \
  DATA.PATH_PREFIX ${DATASET_PATH} \
  DATA.SAMPLING_METHOD ${SAMPLING_METHOD} \
  DATA.PATH_LABEL_SEPARATOR \" \" \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 5 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 5 \
  SOLVER.BASE_LR 1e-5 \
  SOLVER.WARMUP_EPOCHS 1 \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.ADD_SOFTMAX True \
  MODEL.LOSS_FUNC \"focal_loss\" \
  MODEL.FOCAL_LOSS.ENABLE True \
  MODEL.FOCAL_LOSS.ALPHA ${FOCAL_ALPHA} \
  MODEL.FOCAL_LOSS.GAMMA ${FOCAL_GAMMA} \
  OUTPUT_DIR $OUTPUT_DIR"

# Add augmentation parameters if enabled
if [ "$AUGMENTATION_ENABLE" = true ]; then
  CMD="$CMD \
  DATA.AUGMENTATION.ENABLE True \
  DATA.AUGMENTATION.METHOD ${AUGMENTATION_METHOD}"
  
  if [ -n "$MAX_AUG_ROUNDS" ]; then
    CMD="$CMD \
    DATA.AUGMENTATION.MAX_ROUNDS ${MAX_AUG_ROUNDS}"
  fi
fi

# Execute the command
eval $CMD