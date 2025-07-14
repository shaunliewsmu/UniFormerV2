#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from torchvision import transforms
import math
import numpy as np

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics_sparse(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
            cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
            cfg.TEST.NUM_SPATIAL_CROPS = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        
        # Setup data augmentation parameters
        self.augmentation_enabled = (
            cfg.DATA.AUGMENTATION.ENABLE and self.mode == "train"
        )
        
        if self.augmentation_enabled:
            logger.info("Data augmentation enabled for training")
            self.sampling_method = cfg.DATA.AUGMENTATION.METHOD
            self.max_aug_rounds = cfg.DATA.AUGMENTATION.MAX_ROUNDS
            self.aug_step_size = cfg.DATA.AUGMENTATION.STEP_SIZE  # Get step size from config
            logger.info(f"Using sampling method: {self.sampling_method}")
            logger.info(f"Using augmentation step size: {self.aug_step_size}")
            if self.max_aug_rounds:
                logger.info(f"Maximum augmentation rounds: {self.max_aug_rounds}")
            else:
                logger.info("Maximum augmentation rounds will be auto-calculated based on video length")
            
            # Setup data augmentation information for all videos
            self._setup_augmentation()
        else:
            # For regular sampling method (without augmentation)
            if hasattr(cfg.DATA, 'SAMPLING_METHOD'):
                self.sampling_method = cfg.DATA.SAMPLING_METHOD
                logger.info(f"Using sampling method: {self.sampling_method} for {mode} mode")
            else:
                self.sampling_method = "uniform"
                logger.info(f"No sampling method specified, using default: {self.sampling_method}")

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def _setup_augmentation(self):
        """Initialize augmentation information for all videos in the dataset."""
        logger.info(f"Setting up data augmentation with {self.sampling_method} sampling method")
        
        # Store original dataset length before augmentation
        self.original_length = len(self._path_to_videos)
        
        # Mappings for augmented samples
        self.aug_video_map = []  # Maps augmented index to (video_idx, aug_round)
        self.augmented_labels = []  # Store labels for all augmented samples
        
        # Process each video to calculate augmentation rounds and store mappings
        total_augmented_samples = 0
        
        # Count samples for each class before augmentation
        class_counts_before = {}
        for label in self._labels:
            class_counts_before[label] = class_counts_before.get(label, 0) + 1
        
        # Create augmentation mappings and count augmented samples per class
        for video_idx, video_path in enumerate(self._path_to_videos):
            # Get video frame count
            cap = None
            try:
                video_container = container.get_video_container(
                    video_path,
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
                if hasattr(video_container, 'streams') and hasattr(video_container.streams, 'video'):
                    total_frames = video_container.streams.video[0].frames
                else:
                    # For backends that don't provide frame count directly, we need to estimate
                    total_frames = 300  # Default estimate, can be refined
            except Exception as e:
                logger.warning(f"Could not open video {video_path} to get frame count: {e}")
                total_frames = 300  # Default estimate if we can't open the video
            
            # Calculate max augmentation rounds for this video
            if self.max_aug_rounds is None:
                video_max_rounds = self._calculate_max_aug_rounds(total_frames, self.cfg.DATA.NUM_FRAMES)
            else:
                video_max_rounds = min(self.max_aug_rounds, 
                                     self._calculate_max_aug_rounds(total_frames, self.cfg.DATA.NUM_FRAMES))
            
            # Get the video's label
            label = self._labels[video_idx]
            
            # Add mapping entries for augmented samples
            for aug_round in range(1, video_max_rounds + 1):
                self.aug_video_map.append((video_idx, aug_round))
                self.augmented_labels.append(label)
                total_augmented_samples += 1
        
        # Calculate class counts after augmentation
        class_counts_after = class_counts_before.copy()
        for label in self.augmented_labels:
            class_counts_after[label] = class_counts_after.get(label, 0) + 1
        
        # Log augmentation info
        logger.info(f"Data augmentation added {total_augmented_samples} samples to the original {self.original_length}")
        
        # Create class name mapping for nicer output
        class_names = {0: "non-referral", 1: "referral"}
        
        # Calculate and log class distribution before augmentation
        original_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v}" for k, v in sorted(class_counts_before.items())])
        logger.info(f"Original class distribution: {original_dist_str}")
        
        # Calculate and log class distribution after augmentation
        augmented_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v}" for k, v in sorted(class_counts_after.items())])
        logger.info(f"Augmented class distribution: {augmented_dist_str}")
        
        # Calculate added samples per class
        added_counts = {k: class_counts_after.get(k, 0) - class_counts_before.get(k, 0) for k in set(class_counts_before) | set(class_counts_after)}
        added_dist_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: +{v}" for k, v in sorted(added_counts.items())])
        logger.info(f"Added samples per class: {added_dist_str}")
        
        # Calculate class distribution percentages
        original_total = sum(class_counts_before.values())
        original_pct = {k: (v / original_total) * 100 for k, v in class_counts_before.items()}
        original_pct_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v:.1f}%" for k, v in sorted(original_pct.items())])
        
        augmented_total = sum(class_counts_after.values())
        augmented_pct = {k: (v / augmented_total) * 100 for k, v in class_counts_after.items()}
        augmented_pct_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v:.1f}%" for k, v in sorted(augmented_pct.items())])
        
        logger.info(f"Original class percentages: {original_pct_str}")
        logger.info(f"Augmented class percentages: {augmented_pct_str}")
        
        # Calculate augmentation factor per class
        aug_factor = {k: class_counts_after.get(k, 0) / class_counts_before.get(k, 1) for k in class_counts_before}
        aug_factor_str = ", ".join([f"{class_names.get(k, f'class_{k}')}: {v:.2f}x" for k, v in sorted(aug_factor.items())])
        logger.info(f"Augmentation factor per class: {aug_factor_str}")
    
    def _calculate_max_aug_rounds(self, total_frames, num_frames):
        """
        Calculate the maximum number of augmentation rounds based on uniform sampling.
        
        Args:
            total_frames: Total number of frames in the video
            num_frames: Number of frames to sample per round
            
        Returns:
            max_rounds: Maximum number of augmentation rounds
        """
        # Edge case: if num_frames is too large relative to total_frames,
        # the chunks will be too small for meaningful augmentation
        if num_frames > total_frames / 2:
            # When requesting too many frames, limit augmentation rounds
            return max(1, min(5, total_frames // (2 * num_frames) + 1))
        
        # For uniform sampling with original method:
        if num_frames <= 1:
            return 1  # No augmentation possible with only 1 frame
        
        # Calculate step using original formula
        step = (total_frames - 1) / (num_frames - 1)
        
        # Find the minimum space between consecutive frames - this determines max rounds
        min_chunk_size = max(1, int(step) - 1)  # At least 1 frame between borders
        
        # Each augmentation round uses one frame from each chunk
        # Divide by step size to get the effective number of rounds
        max_rounds = max(1, min_chunk_size // self.aug_step_size)
        
        return max_rounds

    def __len__(self):
        """
        Return the total length of the dataset including augmentations.
        """
        if self.augmentation_enabled:
            return self.original_length + len(self.aug_video_map)
        return len(self._path_to_videos)

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        # Handle augmented indices - this is the key data augmentation part
        aug_round = None
        if self.augmentation_enabled and index >= self.original_length:
            # This is an augmented sample
            aug_idx = index - self.original_length
            video_idx, aug_round = self.aug_video_map[aug_idx]
            index = video_idx

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = ([self.cfg.DATA.TEST_CROP_SIZE] * 3)
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                elif self.mode in ["test"] and i_try > self._num_retries // 2:
                    # BUG: should not repeat video
                    logger.info(
                        "Failed to load video idx {} from {}; use idx {}".format(
                            index, self._path_to_videos[index], index - 1
                        )
                    )
                    index = index - 1
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                    video_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,
                    use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                    sparse=True,
                    sampling_method=self.sampling_method,
                    aug_round=aug_round
                )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # After successfully decoding frames
            # Save visualization for the first few videos in each epoch
            if index < 3 and self.mode == "train":  # Only visualize first 3 videos to avoid too many images
                # Import the necessary modules
                import os
                import matplotlib.pyplot as plt
                
                # Create output directory if it doesn't exist
                output_dir = os.path.join(os.path.dirname(self.cfg.OUTPUT_DIR), "sampled_frames")
                os.makedirs(output_dir, exist_ok=True)
                
                # Convert frames to numpy for visualization
                frames_for_vis = frames[0].cpu().numpy() if isinstance(frames, list) else frames.cpu().numpy()
                
                # Visualize the frames
                try:
                    # Create a figure with subplots for each frame
                    fig, axes = plt.subplots(1, min(8, frames_for_vis.shape[0]), figsize=(20, 4))
                    
                    # If only one frame, wrap axes in a list
                    if frames_for_vis.shape[0] == 1:
                        axes = [axes]
                    
                    # Loop through frames and display each one
                    for i, ax in enumerate(axes):
                        if i < frames_for_vis.shape[0]:
                            # Convert from channels-first to channels-last for display
                            frame = frames_for_vis[i].transpose(1, 2, 0)
                            
                            # Normalize frame for display
                            frame = (frame - frame.min()) / (frame.max() - frame.min())
                            
                            # Display the frame
                            ax.imshow(frame)
                            ax.set_title(f"Frame {i}")
                            ax.axis('off')
                    
                    # Create filename with video info
                    video_name = os.path.basename(self._path_to_videos[index])
                    aug_info = f"_aug{aug_round}" if aug_round is not None else ""
                    vis_path = os.path.join(output_dir, f"{video_name}_{self.sampling_method}{aug_info}.png")
                    
                    # Save the figure
                    plt.tight_layout()
                    plt.savefig(vis_path)
                    plt.close(fig)
                    logger.info(f"Visualized sampled frames: {vis_path}")
                except Exception as e:
                    logger.error(f"Failed to visualize frames: {e}")
            
            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:

                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        label = self._labels[index]
                        new_frames = utils.pack_pathway_output(
                            self.cfg, new_frames
                        )
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(
                        frames,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )

            else:
                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to load video idx {} from {} after {} retries".format(
                    index, self._path_to_videos[index], self._num_retries
                )
            )

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)