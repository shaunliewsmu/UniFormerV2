# UniFormerV2 Training Guide for Custom Datasets

This guide covers how to set up and train the UniFormerV2 model on a custom binary classification dataset. UniFormerV2 is a powerful video classification framework that combines Vision Transformers with efficient designs.

## 1. Installation

### 1.1 Create and activate a Python virtual environment

```bash
# Create a virtual environment
python -m venv uniformer_env

# Activate the virtual environment
# On Windows:
uniformer_env\Scripts\activate
# On macOS/Linux:
source uniformer_env/bin/activate
```

### 1.2 Install dependencies

```bash
# Core packages
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install numpy torch>=1.9 torchvision
pip install simplejson
pip install iopath
pip install psutil
pip install opencv-python
pip install tensorboard
pip install pytorchvideo
pip install decord

# Additional dependencies
pip install av
pip install pyyaml  # Will be installed with fvcore
pip install tqdm  # Will be installed with fvcore
pip install moviepy  # Optional, for visualizing videos on tensorboard
pip install timm

# Replace sklearn with scikit-learn in setup.py if needed
pip install scikit-learn
```

### 1.3 Clone and install UniFormerV2

```bash
# Clone the repository
git clone https://github.com/OpenGVLab/UniFormerV2.git
cd UniFormerV2

# Modify setup.py if needed (replace "sklearn" with "scikit-learn")

# Build and install the package
python setup.py build develop
```

## 2. Prepare the CLIP pre-trained model

### 2.1 Extract the CLIP visual encoder

Create a Python script to extract the CLIP visual encoder:

```python
# extract_clip.py
import torch
from collections import OrderedDict

# Import CLIP (ensure you have clip installed: pip install git+https://github.com/openai/CLIP.git)
import clip

# Load the model
model, _ = clip.load("ViT-B/16", device='cpu')

# Extract the visual encoder
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v

# Save the extracted model
torch.save(new_state_dict, 'extract_clip/vit_b16.pth')
```

Run the script:

```bash
mkdir -p extract_clip
python extract_clip.py
```

### 2.2 Update the model path

Edit `slowfast/models/uniformerv2_model.py` to update the `MODEL_PATH` variable:

```python
# Change this line
MODEL_PATH = '/absolute/path/to/your/UniFormerV2/extract_clip'
```

## 3. Prepare your custom dataset

### 3.1 Create directory structure

```bash
mkdir -p data_list/custom_referral
```

### 3.2 Generate CSV files for the dataset

Create a Python script to generate CSV files from your dataset structure:

```python
# generate_csv.py
import os
import csv

# Define paths and labels
base_path = "data/laryngeal_dataset_balanced/dataset"  # Update to your dataset path
splits = ["train", "val", "test"]
classes = {"non_referral": 0, "referral": 1}

# Path to save CSV files
output_dir = "data_list/custom_referral"
os.makedirs(output_dir, exist_ok=True)

# Process each split
for split in splits:
    csv_path = os.path.join(output_dir, f"{split}.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        
        # Process each class
        for class_name, label in classes.items():
            class_dir = os.path.join(base_path, split, class_name)
            
            # Skip if directory doesn't exist
            if not os.path.exists(class_dir):
                continue
            
            # Get all video files in the directory
            for video_file in os.listdir(class_dir):
                if video_file.endswith(('.mp4', '.avi', '.mkv')):
                    video_path = os.path.join(split, class_name, video_file)
                    writer.writerow([video_path, label])
    
    print(f"Created {csv_path}")
```

Run the script:

```bash
python generate_csv.py
```

## 4. Create experiment configuration

### 4.1 Create the experiment directory

```bash
mkdir -p exp/custom_referral/custom_b16_f8x224
```

### 4.2 Create config.yaml file

```bash
cat > exp/custom_referral/custom_b16_f8x224/config.yaml << 'EOF'
TRAIN:
  ENABLE: True
  DATASET: kinetics_sparse
  BATCH_SIZE: 64
  EVAL_PERIOD: 1
  CHECKPOINT_PERIOD: 5
  AUTO_RESUME: True
DATA:
  USE_OFFSET_SAMPLING: True
  DECODING_BACKEND: decord
  NUM_FRAMES: 8
  SAMPLING_RATE: 16
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 224
  INPUT_CHANNEL_NUM: [3]
  PATH_LABEL_SEPARATOR: " "
UNIFORMERV2:
  BACKBONE: 'uniformerv2_b16'
  N_LAYERS: 4
  N_DIM: 768
  N_HEAD: 12
  MLP_FACTOR: 4.0
  BACKBONE_DROP_PATH_RATE: 0.
  DROP_PATH_RATE: 0.
  MLP_DROPOUT: [0.5, 0.5, 0.5, 0.5]
  CLS_DROPOUT: 0.5
  RETURN_LIST: [8, 9, 10, 11]
  NO_LMHRA: True
  TEMPORAL_DOWNSAMPLE: False
MODEL:
  NUM_CLASSES: 2
  ARCH: uniformerv2
  MODEL_NAME: Uniformerv2
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
TEST:
  ENABLE: True
  DATASET: kinetics_sparse
  BATCH_SIZE: 64
  NUM_SPATIAL_CROPS: 3
  NUM_ENSEMBLE_VIEWS: 4
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
SOLVER:
  BASE_LR: 1e-5
  WARMUP_EPOCHS: 1.0
  MAX_EPOCH: 50
  OPTIMIZING_METHOD: adamw
  WEIGHT_DECAY: 0.05
  MOMENTUM: 0.9
NUM_GPUS: 0
OUTPUT_DIR: ./output/custom_referral
EOF
```

### 4.3 Create training script

```bash
cat > exp/custom_referral/custom_b16_f8x224/run.sh << 'EOF'
#!/bin/bash

work_path=$(dirname $0)
NUM_SHARDS=1
NUM_GPUS=0  # Set to 0 for non-distributed training
BATCH_SIZE=8

PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
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
  SOLVER.WARMUP_EPOCHS 1 \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.ADD_SOFTMAX True \
  OUTPUT_DIR $work_path/output
EOF
```

### 4.4 Create testing script

```bash
cat > exp/custom_referral/custom_b16_f8x224/test.sh << 'EOF'
#!/bin/bash

work_path=$(dirname $0)
NUM_SHARDS=1
NUM_GPUS=0  # Set to 0 for non-distributed training
BATCH_SIZE=8

PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg $work_path/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR ./data_list/custom_referral \
  DATA.PATH_PREFIX data/laryngeal_dataset_balanced/dataset \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.CHECKPOINT_FILE_PATH $work_path/output/checkpoints/checkpoint_best.pyth \
  TEST.ADD_SOFTMAX True \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  OUTPUT_DIR $work_path/output
EOF
```

### 4.5 Make the scripts executable

```bash
chmod +x exp/custom_referral/custom_b16_f8x224/run.sh
chmod +x exp/custom_referral/custom_b16_f8x224/test.sh
```

## 5. Train the model

```bash
./exp/custom_referral/custom_b16_f8x224/run.sh
```

## 6. Test the model

After training completes, test the model:

```bash
./exp/custom_referral/custom_b16_f8x224/test.sh
```

## 7. Monitoring and Visualization

Enable TensorBoard visualization by setting `TENSORBOARD.ENABLE: True` in your config file. Then run:

```bash
tensorboard --logdir exp/custom_referral/custom_b16_f8x224/output
```

## Troubleshooting

### Common issues and solutions:

1. **Path errors**: Make sure all paths in the scripts point to the correct locations.

2. **GPU memory issues**: If you encounter GPU memory problems:
   - Reduce the batch size
   - Reduce the number of frames
   - Use a smaller model

3. **Distributed training issues**: 
   - If you want to use distributed training (multiple GPUs), set `NUM_GPUS` to the number of GPUs and use `--init_method tcp://localhost:10125` parameter
   - For simple single-GPU training, set `NUM_GPUS: 0` to avoid distributed training initialization

4. **CUDA errors**: Make sure PyTorch is installed with CUDA support that matches your CUDA version:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
   ```

5. **Path with special characters**: If your dataset path contains special characters like colons (`:`), create a symbolic link:
   ```bash
   ln -s path/with/special:characters/dataset clean_path
   ```

6. **Type mismatch errors**: Make sure the data types in your command line arguments match those expected in the configuration. For example, use integers instead of floats if the configuration expects integers.

## Additional Resources

- Official UniFormerV2 GitHub repository: [https://github.com/OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)
- PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- CLIP GitHub repository: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)