#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
import os
import json
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from timm.utils import NativeScaler

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint_amp as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from iopath.common.file_io import g_pathmgr
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score,
                            average_precision_score,precision_score,recall_score, accuracy_score)

logger = logging.get_logger(__name__)


def calculate_class_weights(labels):
    """
    Calculate class weights inversely proportional to class frequency.
    
    Args:
        labels (torch.Tensor): Training labels
        
    Returns:
        torch.Tensor: Class weights for loss function
    """
    import numpy as np
    
    # Convert to numpy for easier handling
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Count class occurrences    
    class_counts = np.bincount(labels)
    
    # Inverse frequency weighting
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Normalize weights so they sum to 1 * num_classes
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return torch.tensor(class_weights, dtype=torch.float32)


def calculate_metrics(all_labels, all_preds, all_scores=None):
    """
    Calculate comprehensive metrics for classification.
    
    Args:
        all_labels (array): Ground truth labels
        all_preds (array): Predicted class indices
        all_scores (array): Prediction scores/probabilities for positive class
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Calculate standard metrics
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['precision'] = precision_score(all_labels, all_preds, zero_division=0)
    metrics['recall'] = recall_score(all_labels, all_preds, zero_division=0)
    metrics['f1'] = f1_score(all_labels, all_preds, zero_division=0)
    
    # Calculate ROC-AUC if we have probability scores for binary classification
    if len(np.unique(all_labels)) == 2 and all_scores is not None:
        metrics['auroc'] = roc_auc_score(all_labels, all_scores)
        metrics['auprc'] = average_precision_score(all_labels, all_scores)
        
        # Calculate confusion matrix for binary classification
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2.0
            
            # Calculate diversity (how evenly distributed predictions are)
            total = tn + fp + fn + tp
            neg_pred_ratio = (tn + fn) / total if total > 0 else 0
            pos_pred_ratio = (tp + fp) / total if total > 0 else 0
            # Entropy based diversity - maximum when equal distribution
            epsilon = 1e-10  # Small value to avoid log(0)
            metrics['diversity'] = -(neg_pred_ratio * np.log2(neg_pred_ratio + epsilon) + 
                                    pos_pred_ratio * np.log2(pos_pred_ratio + epsilon))
    
    return metrics


def should_save_model(val_metrics, best_metrics):
    """
    Determine if model should be saved based on a balanced composite score.
    
    Args:
        val_metrics (dict): Current validation metrics
        best_metrics (dict): Best metrics so far
        
    Returns:
        bool: Whether to save this model
        float: Composite score
    """
    # Check if the model is predicting only one class (specificity or recall is 0)
    if val_metrics.get('specificity', 1.0) <= 0.01:
        logger.info(f"Model predicts only one class (specificity: {val_metrics.get('specificity', 0):.4f})")
        if best_metrics.get('specificity', 0) > 0.01:
            logger.info("Not saving - previous model had better class balance")
            return False, 0.0
    
    if val_metrics.get('recall', 1.0) <= 0.01:
        logger.info(f"Model predicts only one class (recall: {val_metrics.get('recall', 0):.4f})")
        if best_metrics.get('recall', 0) > 0.01:
            logger.info("Not saving - previous model had better class balance")
            return False, 0.0
    
    # Calculate composite score - weighted average of different metrics
    current_score = (
        (0.20 * val_metrics.get('f1', 0)) +              # 20% weight on F1
        (0.15 * val_metrics.get('auroc', 0)) +           # 15% weight on AUROC
        (0.15 * val_metrics.get('auprc', 0)) +           # 15% weight on AUPRC
        (0.25 * val_metrics.get('balanced_accuracy', 0)) + # 25% weight on balanced accuracy
        (0.25 * val_metrics.get('diversity', 0))         # 25% weight on diversity
    )
    
    best_score = best_metrics.get('composite_score', -float('inf'))
    
    # Only save if it's better than the previous best
    if current_score > best_score:
        logger.info(f"New best model with composite score: {current_score:.4f} (previous: {best_score:.4f})")
        return True, current_score
    
    return False, current_score


def train_epoch(
    train_loader, model, optimizer, loss_scaler, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        loss_scaler (scaler): scaler for loss.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast():
            if cfg.DETECTION.ENABLE:
                preds = model(inputs, meta["boxes"])
            else:
                preds = model(inputs)
                
            # Add debugging info
            if cur_iter == 0 or cur_iter % 20 == 0:
                logger.info(f"Iteration {cur_iter}: Preds min: {preds.min().item()}, "
                           f"max: {preds.max().item()}, has_nan: {torch.isnan(preds).any().item()}")
                logger.info(f"Labels: {labels.detach().cpu().numpy()}, "
                          f"has_nan: {torch.isnan(labels).any().item()}")
            
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC, cfg)(reduction="mean")

            # Compute the loss.
            loss = loss_fun(preds, labels)
            
            if cur_iter == 0 or cur_iter % 20 == 0:
                logger.info(f"Iteration {cur_iter}: Loss: {loss.item()}, is_nan: {torch.isnan(loss).item()}")

        # Check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        # scaler => backward and step
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=cfg.SOLVER.CLIP_GRADIENT, parameters=model.parameters(), create_graph=is_second_order)
        
        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, min(5, cfg.MODEL.NUM_CLASSES)))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, loss_scaler, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        loss_scaler (scaler): scaler for loss.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            # Add softmax for evaluation if needed
            if cfg.TEST.ADD_SOFTMAX:
                preds = model(inputs).softmax(-1)
            else:
                preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, min(5, cfg.MODEL.NUM_CLASSES)))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Gather all predictions and labels for computing metrics
    all_preds = []
    all_labels = []
    
    for pred, label in zip(val_meter.all_preds, val_meter.all_labels):
        all_preds.append(pred.clone().detach())
        all_labels.append(label.clone().detach())
    
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    
    # Calculate comprehensive metrics
    pred_classes = np.argmax(all_preds, axis=1)
    
    # Calculate scores for binary classification (positive class probability)
    if cfg.MODEL.NUM_CLASSES == 2:
        pos_scores = all_preds[:, 1]
        metrics_dict = calculate_metrics(all_labels, pred_classes, pos_scores)
        
        # Log comprehensive metrics
        logger.info("\n" + "="*50)
        logger.info("VALIDATION METRICS:")
        logger.info("="*50)
        for key, value in metrics_dict.items():
            if key != 'confusion_matrix':
                logger.info(f"{key}: {value:.4f}")
        
        # Log confusion matrix
        if 'confusion_matrix' in metrics_dict:
            cm = metrics_dict['confusion_matrix']
            logger.info(f"Confusion Matrix:\n{cm}")
        
        # Log to tensorboard
        if writer is not None:
            for key, value in metrics_dict.items():
                if key != 'confusion_matrix' and isinstance(value, (int, float)):
                    writer.add_scalar(f"Val/{key}", value, global_step=cur_epoch)
    
    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    
    # Save comprehensive metrics for model selection
    metrics_to_return = {}
    if cfg.MODEL.NUM_CLASSES == 2:
        metrics_to_return = metrics_dict
    
    # Reset the meter for next epoch
    val_meter.reset()
    
    return metrics_to_return


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Configure loss function - including focal loss support
    if hasattr(cfg.MODEL, 'FOCAL_LOSS') and cfg.MODEL.FOCAL_LOSS.ENABLE:
        logger.info(f"Using Focal Loss with alpha={cfg.MODEL.FOCAL_LOSS.ALPHA}, gamma={cfg.MODEL.FOCAL_LOSS.GAMMA}")
        cfg.MODEL.LOSS_FUNC = "focal_loss"
    
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Loss scaler
    loss_scaler = NativeScaler()

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        loss_scaler,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Ensure output directory exists
    if not g_pathmgr.exists(cfg.OUTPUT_DIR):
        g_pathmgr.mkdirs(cfg.OUTPUT_DIR)
        logger.info(f"Created output directory: {cfg.OUTPUT_DIR}")

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Check for focal loss configuration
    if hasattr(cfg.MODEL, 'FOCAL_LOSS') and cfg.MODEL.FOCAL_LOSS.ENABLE:
        logger.info(f"Using Focal Loss with alpha={cfg.MODEL.FOCAL_LOSS.ALPHA}, gamma={cfg.MODEL.FOCAL_LOSS.GAMMA}")
        
        # Check for auto alpha
        if hasattr(cfg.MODEL.FOCAL_LOSS, 'AUTO_ALPHA') and cfg.MODEL.FOCAL_LOSS.AUTO_ALPHA:
            logger.info("Auto Alpha enabled - will calculate optimal alpha from class distribution")

    # Loss scaler
    loss_scaler = NativeScaler()

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, loss_scaler)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    logger.info(f"Training in directory: {cfg.OUTPUT_DIR}")

    # Save a copy of the config file to the output directory
    config_save_path = os.path.join(cfg.OUTPUT_DIR, "config_used.yaml")
    with g_pathmgr.open(config_save_path, "w") as f:
        f.write(str(cfg))
    logger.info(f"Config saved to {config_save_path}")

    # Extract sampling method from the path if available
    if hasattr(cfg.DATA, 'SAMPLING_METHOD') and cfg.DATA.SAMPLING_METHOD:
        sampling_method = cfg.DATA.SAMPLING_METHOD
        logger.info(f"Using sampling method: {sampling_method}")
    else:
        sampling_method = "unknown"
        logger.info("Sampling method not specified in config")

    # Save sampling method to a file for reference
    sampling_method_path = os.path.join(cfg.OUTPUT_DIR, "sampling_method.txt")
    with g_pathmgr.open(sampling_method_path, "w") as f:
        f.write(f"Sampling method: {sampling_method}\n")

    # Initialize dictionary for tracking best metrics
    best_metrics = {
        'val_loss': float('inf'),
        'val_acc': 0.0,
        'val_auroc': 0.0,
        'val_auprc': 0.0,
        'val_f1': 0.0,
        'val_precision': 0.0,
        'val_recall': 0.0,
        'val_specificity': 0.0,
        'val_balanced_accuracy': 0.0,
        'val_diversity': 0.0,
        'composite_score': -float('inf'),
        'epoch': -1
    }

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    loss_scaler,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader, model, optimizer, loss_scaler, train_meter, cur_epoch, cfg, writer
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)
            logger.info(f"Regular checkpoint saved for epoch {cur_epoch}")
            
        # Evaluate the model on validation set.
        if is_eval_epoch:
            logger.info(f"Evaluating epoch {cur_epoch}")
            val_metrics = eval_epoch(val_loader, model, val_meter, loss_scaler, cur_epoch, cfg, writer)
            
            # For binary classification, check if we should save the model based on balanced metrics
            if cfg.MODEL.NUM_CLASSES == 2 and hasattr(cfg.TEST, 'METRICS') and hasattr(cfg.TEST.METRICS, 'USE_BALANCED_METRICS') and cfg.TEST.METRICS.USE_BALANCED_METRICS:
                # Check if metrics were returned
                if val_metrics:
                    save_model, composite_score = should_save_model(val_metrics, best_metrics)
                    
                    # Update current metrics with composite score
                    val_metrics['composite_score'] = composite_score
                    
                    if save_model:
                        best_metrics = val_metrics.copy()
                        best_metrics['epoch'] = cur_epoch
                        
                        # Save as best checkpoint
                        cu.save_checkpoint(
                            cfg.OUTPUT_DIR, 
                            model, 
                            optimizer, 
                            loss_scaler, 
                            cur_epoch, 
                            cfg
                        )
                        logger.info(f"Saved new best model at epoch {cur_epoch} with composite score {composite_score:.4f}")
                        
                        # Save best metrics to a file
                        best_metrics_path = os.path.join(cfg.OUTPUT_DIR, "best_metrics.json")
                        with open(best_metrics_path, 'w') as f:
                            # Convert numpy values to Python native types for JSON
                            best_metrics_json = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                               for k, v in best_metrics.items() if k != 'confusion_matrix'}
                            json.dump(best_metrics_json, f, indent=4)
                        logger.info(f"Best metrics saved to {best_metrics_path}")
            else:
                # Standard model saving logic based on validation loss
                if val_meter.stats["top1_acc"] > best_metrics['val_acc']:
                    best_metrics['val_acc'] = val_meter.stats["top1_acc"]
                    best_metrics['epoch'] = cur_epoch
                    
                    # Save as best checkpoint
                    cu.save_checkpoint(
                        cfg.OUTPUT_DIR, 
                        model, 
                        optimizer, 
                        loss_scaler, 
                        cur_epoch, 
                        cfg
                    )
                    logger.info(f"Saved new best model at epoch {cur_epoch} with accuracy {best_metrics['val_acc']:.4f}")

        # Save latest model
        if cfg.TRAIN.SAVE_LATEST:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)
            logger.info(f"Latest checkpoint saved for epoch {cur_epoch}")

    logger.info(f"Training completed in directory: {cfg.OUTPUT_DIR}")
    
    # Log final best metrics
    logger.info("\nBest model performance:")
    logger.info(f"Best epoch: {best_metrics['epoch']}")
    
    for key, value in best_metrics.items():
        if key not in ['epoch', 'confusion_matrix'] and isinstance(value, (int, float, np.number)):
            logger.info(f"{key}: {value:.4f}")
    
    if writer is not None:
        writer.close()