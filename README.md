# UniFormerV2 Video Classifier

This is a UniFormerV2 implementation for laryngeal cancer screening that provides advanced video transformer architecture with unified space-time attention mechanisms. UniFormerV2 is implemented in a **separate repository** and uses a different training framework compared to other models in this project.

## Project Overview

This UniFormerV2 implementation offers:
- **Unified space-time attention**: Advanced video transformer with integrated spatial and temporal modeling
- **Multi-scale feature learning**: Hierarchical representation learning for better video understanding
- **Comprehensive evaluation**: Automated generation of ROC curves, PR curves, and confusion matrices
- **Multi-GPU support**: Distributed training capabilities for faster training
- **Flexible configuration**: YAML-based configuration system with command-line overrides

## Repository Setup

### Installation

```bash
git clone https://github.com/mhleerepo/UniFormerV2.git  
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset Installation

All preprocessed datasets are available on the mercury server. Copy the datasets to your cloned repository:

#### Available Datasets
1. **Balanced (Duke + BAGLS) Dataset** (`data/balanced-dataset`)
2. **BAGLS Dataset** (`data/bagls-split:v0`)
3. **Duke Dataset** (`data/duhs-gss-split-5:v0`)

#### Copy Commands

SSH into the mercury server and navigate to `/mnt/storage/shaun/UniFormerV2`, then copy the required folders:

```bash
# Copy data folder
rsync -av /mnt/storage/shaun/UniFormerV2/data/ /path/to/destination/
# OR
cp -r /mnt/storage/shaun/UniFormerV2/data/ /path/to/destination/

# Copy data_list folder (required for training)
rsync -av /mnt/storage/shaun/UniFormerV2/data_list/ /path/to/destination/
# OR
cp -r /mnt/storage/shaun/UniFormerV2/data_list/ /path/to/destination/
```

> **Important**: Place the copied `data` and `data_list` folders under the parent directory of UniFormerV2.

## Project Structure

```
UniFormerV2/
├── README.md                                    # Original Project documentation
├── NEW_README.md                                # Latest Project Documentation
├── DATASET.md                                   # Dataset setup instructions
├── INSTALL.md                                   # Installation instructions
├── INSTRUCTIONS.md                              # Usage instructions
├── MODEL_ZOO.md                                 # Available model configurations
├── requirements.txt                             # Python dependencies
├── setup.py                                     # Package setup
├── generate_csv.py                              # CSV generation utility
├── validate_video.py                            # Video validation script
├── data/                                        # Dataset storage
│   ├── bagls-split:v0/                         # BAGLS dataset
│   ├── balanced-dataset/                       # Balanced Duke+BAGLS dataset
│   ├── combined-dataset-imbalanced/            # Combined imbalanced dataset
│   ├── duhs-gss-split-1:v0/                    # Duke dataset split 1
│   ├── duhs-gss-split-2:v0/                    # Duke dataset split 2
│   ├── duhs-gss-split-3:v0/                    # Duke dataset split 3
│   ├── duhs-gss-split-4:v0/                    # Duke dataset split 4
│   └── duhs-gss-split-5:v0/                    # Duke dataset split 5
├── data_list/                                   # Data list files for training
│   ├── bagls-split/                            # BAGLS data lists
│   ├── balanced-duke-using-duke/               # Balanced dataset lists
│   ├── custom_referral/                        # Custom referral lists
│   └── duke-and-bagls/                         # Combined dataset lists
├── exp/                                         # Experiment configurations
│   ├── custom_referral/                        # Custom referral experiments
│   │   └── custom_b16_f8x224/                  # Main experiment configuration
│   │       ├── config.yaml                     # Model configuration
│   │       ├── train.sh                        # Training script (no augmentation)
│   │       ├── train_aug.sh                    # Training script (with augmentation)
│   │       ├── fine_tune.sh                    # Fine-tuning script
│   │       ├── evaluate.sh                     # Evaluation script
│   │       └── output/                         # Training results
│   │           ├── run_YYYYMMDD_HHMMSS_*/     # Timestamped training runs
│   │           ├── fine_tune_run_*/            # Fine-tuning runs
│   │           └── sampled_frames/             # Sample frame visualizations
│   ├── k400/                                   # Kinetics-400 experiments
│   ├── k600/                                   # Kinetics-600 experiments
│   └── ...                                     # Other dataset experiments
├── slowfast/                                    # Core framework
│   ├── config/                                 # Configuration system
│   ├── datasets/                               # Dataset handling
│   ├── models/                                 # Model implementations
│   ├── utils/                                  # Utility functions
│   └── visualization/                          # Visualization tools
├── tools/                                       # Training and evaluation tools
│   ├── run_net.py                              # Main training/evaluation script
│   ├── test_net.py                             # Testing script
│   ├── train_net.py                            # Training script
│   └── ...                                     # Other utilities
├── models/                                      # Pretrained model weights
│   ├── vit_b16.pth                            # ViT-B16 weights
│   ├── vit_l14.pth                            # ViT-L14 weights
│   └── vit_l14_336.pth                        # ViT-L14-336 weights
└── extract_clip/                                # CLIP feature extraction
    ├── clip.py                                 # CLIP model
    ├── model.py                                # CLIP architecture
    └── ...                                     # CLIP utilities
```

## Training Output Structure

Each training run creates a timestamped directory under `exp/custom_referral/custom_b16_f8x224/output/`:

```
run_YYYYMMDD_HHMMSS_[sampling_method]_alpha[X]_gamma[Y]/
├── 32x224x4x3.pkl                              # Processed data cache
├── best_metrics.json                           # Best performance metrics
├── checkpoints/                                # Model checkpoints
│   └── checkpoint_epoch_XXXXX.pyth            # Epoch checkpoints (the main checkpoint path you have to use for fine tune)
├── config_used.yaml                            # Configuration used for training
├── confusion_matrix.png                        # Confusion matrix visualization
├── metrics.json                                # Training metrics
├── metrics.txt                                 # Metrics in text format
├── pr_curve.png                                # Precision-Recall curve
├── roc_curve.png                               # ROC curve
├── sampling_method.txt                         # Sampling method used
└── stdout.log                                  # Training logs
```

## Training Methods

We have **4 training methods** available for UniFormerV2:

### 1. Without Data Augmentation and Without Fine Tune

Use the `train.sh` script for direct training without augmentation.

**Script Location**: `exp/custom_referral/custom_b16_f8x224/train.sh`

**Key Variables to Configure**:
```bash
SAMPLING_METHOD="uniform"     # Options: uniform, random_window, random
BATCH_SIZE=8                  # Adjust based on GPU memory
FOCAL_ALPHA=1.0              # For BAGLS dataset
FOCAL_GAMMA=4.0              # For BAGLS dataset
# FOCAL_ALPHA=0.25           # For Duke dataset
# FOCAL_GAMMA=2.0            # For Duke dataset

# Dataset Configuration
DATALIST_PATH="./data_list/bagls-split"
DATASET_PATH="data/bagls-split:v0/dataset"
# DATALIST_PATH="./data_list/balanced-duke-using-duke"
# DATASET_PATH="data/balanced-dataset"

SOLVER.MAX_EPOCH=5           # Number of training epochs
```

**Command to Run**:
```bash
bash exp/custom_referral/custom_b16_f8x224/train.sh
```

**Command Line Override Options**:
```bash
bash exp/custom_referral/custom_b16_f8x224/train.sh --alpha 1.0 --gamma 4.0 --sampling_method uniform --batch_size 8
```

### 2. Without Data Augmentation and With Fine Tune

Use the `fine_tune.sh` script for two-stage training without augmentation.

**Script Location**: `exp/custom_referral/custom_b16_f8x224/fine_tune.sh`

**Key Variables to Configure**:
```bash
SAMPLING_METHOD="uniform"     # Sampling method for fine-tuning
BATCH_SIZE=8                  # Batch size
FOCAL_ALPHA=1.0              # Focal loss alpha
FOCAL_GAMMA=4.0              # Focal loss gamma

# Checkpoint Configuration
PRETRAINED_CHECKPOINT="exp/custom_referral/custom_b16_f8x224/output/run_20250411_121502_uniform_alpha0.5_gamma2.5/checkpoints/checkpoint_epoch_00005.pyth"
DELETE_SPECIAL_HEAD=true      # Whether to replace classification layer

# Dataset for fine-tuning
DATALIST_PATH="./data_list/bagls-split"
DATASET_PATH="data/bagls-split:v0/dataset"
```

**Process**:
1. First, train using the base dataset to generate a checkpoint
2. Locate the checkpoint in the output directory: `checkpoints/checkpoint_epoch_XXXXX.pyth`
3. Update `PRETRAINED_CHECKPOINT` path in `fine_tune.sh`
4. Run fine-tuning script

**Command to Run**:
```bash
bash exp/custom_referral/custom_b16_f8x224/fine_tune.sh
```

### 3. With Data Augmentation and Without Fine Tune

Use the `train_aug.sh` script for training with data augmentation.

**Script Location**: `exp/custom_referral/custom_b16_f8x224/train_aug.sh`

**Key Variables to Configure**:
```bash
SAMPLING_METHOD="random"          # Should match AUGMENTATION_METHOD
BATCH_SIZE=8                      # Batch size
FOCAL_ALPHA=1.0                   # Focal loss alpha
FOCAL_GAMMA=4.0                   # Focal loss gamma

# Augmentation Configuration
AUGMENTATION_ENABLE=true          # Enable data augmentation
AUGMENTATION_METHOD="random"      # Should match SAMPLING_METHOD
AUG_STEP_SIZE=16                 # Optimal value for controlling augmentation rounds

# Dataset Configuration
DATALIST_PATH="./data_list/balanced-duke-using-duke"
DATASET_PATH="data/balanced-dataset"
```

**Important**: `AUGMENTATION_METHOD` should match `SAMPLING_METHOD` for consistency.

**Command to Run**:
```bash
bash exp/custom_referral/custom_b16_f8x224/train_aug.sh
```

**Command Line Override Options**:
```bash
bash exp/custom_referral/custom_b16_f8x224/train_aug.sh --augmentation_method random --aug_step_size 16
```

### 4. With Data Augmentation and With Fine Tune

> **Status**: Not implemented yet. This is the next development step that needs to be completed.

## Parameter Customization

### Sampling Methods and Data Augmentation Theory

you can find more details about the theory used for sampling methods and data augmentation from [here](https://github.com/mhleerepo/ai-laryngeal-video-based-classifier/blob/main/README_Techniques_Explaination.md)

### Sampling Methods

Available sampling methods:
- **uniform**: Uniformly sample frames across video
- **random**: Randomly sample frames
- **random_window**: Random sampling within windows

### Dataset-Specific Focal Loss Parameters

**For BAGLS Dataset**:
```bash
FOCAL_ALPHA=1.0
FOCAL_GAMMA=4.0
```

**For Duke Dataset**:
```bash
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0
```

### Core Training Parameters
- `BATCH_SIZE`: Training batch size (default: 8)
- `SOLVER.MAX_EPOCH`: Number of training epochs (default: 5)
- `SOLVER.BASE_LR`: Base learning rate (default: 1e-5)
- `SOLVER.WARMUP_EPOCHS`: Warmup epochs (default: 1)
- `NUM_GPUS`: Number of GPUs for distributed training (default: 2)

### Data Augmentation Parameters
- `AUGMENTATION_ENABLE`: Enable/disable data augmentation
- `AUGMENTATION_METHOD`: Must match sampling method
- `AUG_STEP_SIZE`: Control augmentation rounds (optimal: 16)
- `MAX_AUG_ROUNDS`: Maximum augmentation rounds (optional)

### Model Configuration
- `TEST.NUM_ENSEMBLE_VIEWS`: Number of ensemble views (default: 4)
- `TEST.NUM_SPATIAL_CROPS`: Number of spatial crops (default: 3)
- `MODEL.LOSS_FUNC`: Loss function type (focal_loss)

## Command Line Arguments

All training scripts support command line argument overrides:

### Common Arguments
- `--alpha`: Override focal loss alpha parameter
- `--gamma`: Override focal loss gamma parameter
- `--sampling_method`: Override sampling method
- `--batch_size`: Override batch size

### Fine-tuning Specific Arguments
- `--pretrained`: Override checkpoint path
- `--keep_head`: Keep classification layer (don't delete)

### Augmentation Specific Arguments
- `--augmentation`: Enable augmentation
- `--augmentation_method`: Override augmentation method
- `--aug_step_size`: Override augmentation step size
- `--max_aug_rounds`: Override maximum augmentation rounds

## Model Features

### UniFormerV2 Architecture Advantages
- **Unified Space-Time Attention**: Integrated spatial and temporal modeling
- **Multi-Scale Feature Learning**: Hierarchical representation for better video understanding
- **Computational Efficiency**: Optimized attention mechanisms for video processing
- **Transfer Learning**: Leverages pretrained ViT models (B16, L14, L14-336)
- **Scalability**: Multi-GPU distributed training support

### Training Framework Features
- **YAML Configuration**: Comprehensive configuration system
- **Automatic Logging**: Complete training metrics and visualizations
- **Checkpoint Management**: Automatic checkpoint saving and loading
- **Evaluation Tools**: Built-in ROC curves, PR curves, confusion matrices
- **Command Line Flexibility**: Override any parameter from command line

## Important Notes

### Dataset Requirements
- Both `data` and `data_list` folders are required for training
- `data_list` contains the file paths and labels needed by the training framework
- Ensure correct dataset and data_list path pairing

### Checkpoint Management
- Checkpoints are saved in timestamped directories under `output/`
- Use the full path to checkpoint files when fine-tuning
- `DELETE_SPECIAL_HEAD=true` replaces the classification layer during fine-tuning

### Output Analysis
- All training runs generate comprehensive visualizations automatically
- Check `best_metrics.json` for optimal performance metrics
- Use `stdout.log` for detailed training progress
- Confusion matrices and curves are automatically generated

### Development Status
- Methods 1-3 are fully implemented and tested
- Method 4 (Data Aug + Fine Tune) is planned but not yet implemented
- All scripts support flexible parameter overrides

### Best Practices
- Use appropriate focal loss parameters for your dataset
- Match `SAMPLING_METHOD` with `AUGMENTATION_METHOD` when using augmentation
- Start with default parameters and adjust based on results
- Monitor GPU memory usage with distributed training
- Use `AUG_STEP_SIZE=16` as the optimal augmentation control parameter

### Training Results Location
All training results are automatically saved to:
`exp/custom_referral/custom_b16_f8x224/output/`

Each run creates a unique timestamped directory with complete training artifacts, metrics, and visualizations.