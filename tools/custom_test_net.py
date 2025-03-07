#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model with additional metrics."""

import numpy as np
import os
import pickle
import torch
from iopath.common.file_io import g_pathmgr
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
import argparse

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            if cfg.TEST.ADD_SOFTMAX:
                preds = model(inputs).softmax(-1)
            else:
                preds = model(inputs)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather(
                    [preds, labels, video_idx]
                )
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        # Calculate additional metrics - F1 score and AUROC for binary classification
        all_preds_numpy = all_preds.numpy()
        all_labels_numpy = all_labels.numpy()
        
        # Get the top-1 accuracy from the test_meter
        # Call finalize_metrics to make sure all stats are calculated
        test_meter.finalize_metrics()
        
        # Make sure we get a numeric value for top1_acc
        if "top1_acc" in test_meter.stats:
            if isinstance(test_meter.stats["top1_acc"], str):
                # If it's a string, try to convert to float
                try:
                    top1_acc = float(test_meter.stats["top1_acc"])
                except ValueError:
                    # If conversion fails, use a default value
                    top1_acc = 0.0
            else:
                # If it's already a number
                top1_acc = test_meter.stats["top1_acc"]
        else:
            # If the stat doesn't exist, calculate it manually or use default
            top1_acc = 0.0
            if hasattr(test_meter, "num_top1_cor") and hasattr(test_meter, "num_videos"):
                top1_acc = test_meter.num_top1_cor / test_meter.num_videos
                
        logger.info(f"Top-1 Accuracy: {top1_acc}")
        
        # For binary classification (assuming index 1 is the positive class)
        if cfg.MODEL.NUM_CLASSES == 2:
            # Convert predictions to class labels
            pred_classes = all_preds_numpy.argmax(axis=1)
            
            # Calculate F1 score
            f1 = f1_score(all_labels_numpy, pred_classes)
            logger.info(f"F1 Score: {f1:.4f}")
            
            # Calculate ROC-AUC
            try:
                # Use the probability of the positive class for ROC curve
                roc_auc = roc_auc_score(all_labels_numpy, all_preds_numpy[:, 1])
                logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
                
                # Plot ROC curve and save it
                fpr, tpr, _ = roc_curve(all_labels_numpy, all_preds_numpy[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                
                roc_save_path = os.path.join(cfg.OUTPUT_DIR, 'roc_curve.png')
                plt.savefig(roc_save_path)
                logger.info(f"ROC curve saved to {roc_save_path}")
                plt.close()
                
            except ValueError as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                roc_auc = None
            
            # Create and plot confusion matrix
            cm = confusion_matrix(all_labels_numpy, pred_classes)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'], 
                       yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            
            cm_save_path = os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix.png')
            plt.savefig(cm_save_path)
            logger.info(f"Confusion matrix saved to {cm_save_path}")
            plt.close()
            
            # Save metrics to a text file
            metrics_path = os.path.join(cfg.OUTPUT_DIR, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Top-1 Accuracy: {top1_acc}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                if roc_auc is not None:
                    f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")
                f.write(f"Confusion Matrix:\n")
                f.write(f"{cm}\n")
            logger.info(f"Metrics saved to {metrics_path}")
        else:
            # For multi-class classification, calculate macro F1 score
            pred_classes = all_preds_numpy.argmax(axis=1)
            f1_macro = f1_score(all_labels_numpy, pred_classes, average='macro')
            f1_weighted = f1_score(all_labels_numpy, pred_classes, average='weighted')
            logger.info(f"Macro F1 Score: {f1_macro:.4f}")
            logger.info(f"Weighted F1 Score: {f1_weighted:.4f}")
            
            # Create and plot confusion matrix
            cm = confusion_matrix(all_labels_numpy, pred_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            
            cm_save_path = os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix.png')
            plt.savefig(cm_save_path)
            logger.info(f"Confusion matrix saved to {cm_save_path}")
            plt.close()
            
            # Save metrics to a text file
            metrics_path = os.path.join(cfg.OUTPUT_DIR, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Top-1 Accuracy: {top1_acc}\n")
                f.write(f"Macro F1 Score: {f1_macro:.4f}\n")
                f.write(f"Weighted F1 Score: {f1_weighted:.4f}\n")
                f.write(f"Confusion Matrix:\n")
                f.write(f"{cm}\n")
            logger.info(f"Metrics saved to {metrics_path}")

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with g_pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    return test_meter


def test(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    logger.info(f"Add softmax after prediction: {cfg.TEST.ADD_SOFTMAX}")

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()

    # Save predictions for further analysis
    file_name = f'{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TEST_CROP_SIZE}x{cfg.TEST.NUM_ENSEMBLE_VIEWS}x{cfg.TEST.NUM_SPATIAL_CROPS}.pkl'
    with g_pathmgr.open(os.path.join(
        cfg.OUTPUT_DIR, file_name),
        'wb'
    ) as f:
        result = {
            'video_preds': test_meter.video_preds.cpu().numpy(),
            'video_labels': test_meter.video_labels.cpu().numpy()
        }
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a video classification model with additional metrics"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_8x8_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg = assert_and_infer_cfg(cfg)
    
    test(cfg)