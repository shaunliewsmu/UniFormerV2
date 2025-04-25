#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import random
import torch
import torchvision.io as io


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_start_end_idx(
    video_size, clip_size, clip_idx, num_clips, use_offset=False
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


# Add our new function for frame sampling methods
def get_sampling_indices(total_frames, num_frames, sampling_method='uniform', clip_idx=-1, num_clips=1, aug_round=None):
    """
    Get frame indices based on sampling method, handling cases with fewer frames than requested.
    
    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        sampling_method (str): 'uniform', 'random', or 'random_window'
        clip_idx (int): Clip index for test-time sampling
        num_clips (int): Total number of clips to sample
        aug_round (int): Augmentation round for training (None for original sampling)
            
    Returns:
        list: Frame indices to sample
    """
    # For videos with enough frames, use standard sampling
    if total_frames >= num_frames:
        if sampling_method == 'random':
            # For testing mode, we need deterministic sampling based on clip_idx
            if clip_idx >= 0:
                # Set random seed based on clip_idx for reproducibility
                random.seed(42 + clip_idx)
                # Sample a subset of frames based on clip_idx
                segment_size = total_frames / num_clips
                start_idx = int(segment_size * clip_idx)
                end_idx = int(segment_size * (clip_idx + 1))
                # Ensure the range is valid
                end_idx = min(end_idx, total_frames)
                candidate_frames = list(range(start_idx, end_idx))
                # If we don't have enough frames in this segment, sample with replacement
                if len(candidate_frames) < num_frames:
                    indices = sorted(random.choices(candidate_frames, k=num_frames))
                else:
                    indices = sorted(random.sample(candidate_frames, num_frames))
            else:
                # Random sampling without replacement
                indices = sorted(random.sample(range(total_frames), num_frames))
                
        elif sampling_method == 'random_window':
            # Random window sampling
            window_size = total_frames / num_frames
            indices = []
            
            # For testing mode with specific clip_idx
            if clip_idx >= 0:
                random.seed(42 + clip_idx)
                offset = (total_frames - num_frames * window_size) * clip_idx / num_clips
            else:
                offset = 0
                
            for i in range(num_frames):
                start = int(offset + i * window_size)
                end = int(offset + (i + 1) * window_size)
                end = min(end, total_frames)
                end = max(end, start + 1)  # Ensure window has at least 1 frame
                frame_idx = random.randint(start, end - 1)
                indices.append(frame_idx)
                
        else:  # Default to uniform sampling
            # For augmentation, we modify the uniform sampling
            if aug_round is not None and aug_round > 0:
                # Calculate border positions first (these are the same with or without augmentation)
                if num_frames == 1:
                    border_indices = [total_frames // 2]  # Middle frame for single frame
                else:
                    step = (total_frames - 1) / (num_frames - 1)
                    border_indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
                
                # For augmentation, we need to sample frames from each chunk
                num_chunks = len(border_indices) - 1
                round_frames = []
                
                # For each chunk, select one frame based on the augmentation round
                for i in range(num_chunks):
                    chunk_start = border_indices[i]     # Left border
                    chunk_end = border_indices[i + 1]   # Right border
                    
                    # Calculate frame index for this round
                    frame_idx = chunk_start + aug_round
                    
                    # Ensure we stay within the chunk (excluding right border)
                    if frame_idx < chunk_end:
                        round_frames.append(frame_idx)
                    else:
                        # If we run out of frames, use repetition within chunk
                        available_frames = list(range(chunk_start + 1, chunk_end))
                        if available_frames:
                            frame_idx = random.choice(available_frames)
                            round_frames.append(frame_idx)
                        else:
                            # If no frames available, use the left border again
                            round_frames.append(chunk_start)
                
                # Always include the last border for completeness
                if border_indices:
                    round_frames.append(border_indices[-1])
                    
                indices = sorted(round_frames)
            else:
                # Original uniform sampling without augmentation
                if num_frames == 1:
                    # For a single frame, select based on clip_idx
                    if clip_idx >= 0:
                        segment_size = total_frames / num_clips
                        indices = [min(int((clip_idx + 0.5) * segment_size), total_frames - 1)]
                    else:
                        indices = [total_frames // 2]  # Middle frame
                else:
                    # Get start and end indices based on clip_idx
                    if clip_idx >= 0:
                        segment_size = total_frames / num_clips
                        start_idx = int(segment_size * clip_idx)
                        end_idx = int(segment_size * (clip_idx + 1)) - 1
                        # Ensure valid range
                        end_idx = min(end_idx, total_frames - 1)
                    else:
                        start_idx = 0
                        end_idx = total_frames - 1
                        
                    # Sample frames uniformly
                    step = (end_idx - start_idx) / (num_frames - 1) if num_frames > 1 else 0
                    indices = [min(int(start_idx + i * step), total_frames - 1) for i in range(num_frames)]
    
    # For videos with fewer frames than requested, handle accordingly
    else:
        if sampling_method == 'random':
            # With fewer frames, we'll need to allow duplicates
            if clip_idx >= 0:
                random.seed(42 + clip_idx)
            indices = sorted(random.choices(range(total_frames), k=num_frames))
            
        elif sampling_method == 'random_window':
            # For random window with fewer frames, create virtual windows smaller than 1 frame
            indices = []
            window_size = total_frames / num_frames  # Will be < 1
            
            if clip_idx >= 0:
                random.seed(42 + clip_idx)
                
            for i in range(num_frames):
                # Calculate virtual window boundaries
                virtual_start = i * window_size
                virtual_end = (i + 1) * window_size
                
                # Convert to actual frame indices with potential duplicates
                actual_index = min(int(np.floor(virtual_start + (virtual_end - virtual_start) * random.random())), 
                                  total_frames - 1)
                indices.append(actual_index)
                
        else:  # Uniform sampling
            if num_frames == 1:
                indices = [total_frames // 2]  # Middle frame
            else:
                # Create evenly spaced indices that might include duplicates
                step = total_frames / num_frames
                indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
    
    return indices


def get_seq_frames(video_size, num_frames, clip_idx, num_clips, start_index=0, max_frame=-1):
    seg_size = max(0., float(video_size - 1) / num_frames)
    if max_frame == -1:
        max_frame = int(video_size) - 1
    seq = []
    # index from 1, must add 1
    if clip_idx == -1:
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            idx = min(random.randint(start, end) + start_index, max_frame)
            seq.append(idx)
    else:
        duration = seg_size / (num_clips + 1)
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            frame_index = start + int(duration * (clip_idx + 1))
            idx = min(frame_index + start_index, max_frame)
            seq.append(idx)
    return seq


def pyav_decode_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def torchvision_decode(
    video_handle,
    sampling_rate,
    num_frames,
    clip_idx,
    video_meta,
    num_clips=10,
    target_fps=30,
    modalities=("visual",),
    max_spatial_scale=0,
    use_offset=False,
):
    """
    If video_meta is not empty, perform temporal selective decoding to sample a
    clip from the video with TorchVision decoder. If video_meta is empty, decode
    the entire video and update the video_meta.
    Args:
        video_handle (bytes): raw bytes of the video file.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the clip_idx-th video clip.
        video_meta (dict): a dict contains VideoMetaData. Details can be found
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps.
        modalities (tuple): tuple of modalities to decode. Currently only
            support `visual`, planning to support `acoustic` soon.
        max_spatial_scale (int): the maximal resolution of the spatial shorter
            edge size during decoding.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): if True, the entire video was decoded.
    """
    # Convert the bytes to a tensor.
    video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))

    decode_all_video = True
    video_start_pts, video_end_pts = 0, -1
    # The video_meta is empty, fetch the meta data from the raw video.
    if len(video_meta) == 0:
        # Tracking the meta info for selective decoding in the future.
        meta = io._probe_video_from_memory(video_tensor)
        # Using the information from video_meta to perform selective decoding.
        video_meta["video_timebase"] = meta.video_timebase
        video_meta["video_numerator"] = meta.video_timebase.numerator
        video_meta["video_denominator"] = meta.video_timebase.denominator
        video_meta["has_video"] = meta.has_video
        video_meta["video_duration"] = meta.video_duration
        video_meta["video_fps"] = meta.video_fps
        video_meta["audio_timebas"] = meta.audio_timebase
        video_meta["audio_numerator"] = meta.audio_timebase.numerator
        video_meta["audio_denominator"] = meta.audio_timebase.denominator
        video_meta["has_audio"] = meta.has_audio
        video_meta["audio_duration"] = meta.audio_duration
        video_meta["audio_sample_rate"] = meta.audio_sample_rate

    fps = video_meta["video_fps"]
    if (
        video_meta["has_video"]
        and video_meta["video_denominator"] > 0
        and video_meta["video_duration"] > 0
    ):
        # try selective decoding.
        decode_all_video = False
        clip_size = sampling_rate * num_frames / target_fps * fps
        start_idx, end_idx = get_start_end_idx(
            fps * video_meta["video_duration"],
            clip_size,
            clip_idx,
            num_clips,
            use_offset=use_offset,
        )
        # Convert frame index to pts.
        pts_per_frame = video_meta["video_denominator"] / fps
        video_start_pts = int(start_idx * pts_per_frame)
        video_end_pts = int(end_idx * pts_per_frame)

    # Decode the raw video with the tv decoder.
    v_frames, _ = io._read_video_from_memory(
        video_tensor,
        seek_frame_margin=1.0,
        read_video_stream="visual" in modalities,
        video_width=0,
        video_height=0,
        video_min_dimension=max_spatial_scale,
        video_pts_range=(video_start_pts, video_end_pts),
        video_timebase_numerator=video_meta["video_numerator"],
        video_timebase_denominator=video_meta["video_denominator"],
    )

    if v_frames.shape == torch.Size([0]):
        # failed selective decoding
        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1
        v_frames, _ = io._read_video_from_memory(
            video_tensor,
            seek_frame_margin=1.0,
            read_video_stream="visual" in modalities,
            video_width=0,
            video_height=0,
            video_min_dimension=max_spatial_scale,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase_numerator=video_meta["video_numerator"],
            video_timebase_denominator=video_meta["video_denominator"],
        )

    return v_frames, fps, decode_all_video


def pyav_decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx,
    num_clips=10,
    target_fps=30,
    use_offset=False,
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.
    fps = float(container.streams.video[0].average_rate)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            frames_length,
            sampling_rate * num_frames / target_fps * fps,
            clip_idx,
            num_clips,
            use_offset=use_offset,
        )
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(
            container,
            video_start_pts,
            video_end_pts,
            container.streams.video[0],
            {"video": 0},
        )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    return frames, fps, decode_all_video


def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    backend="pyav",
    max_spatial_scale=0,
    use_offset=False,
    sparse=False,
    total_frames=None,
    start_index=0,
    sampling_method="uniform",
    aug_round=None  # Add augmentation round parameter
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video for testing.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
        sparse (bool): if True, use sparse sampling.
        sampling_method (str): frame sampling method ('uniform', 'random', or 'random_window').
        aug_round (int): augmentation round for training (None for original sampling).
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    try:
        if backend == "pyav":
            frames, fps, decode_all_video = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips,
                target_fps,
                use_offset=use_offset,
            )
        elif backend == "torchvision":
            frames, fps, decode_all_video = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips,
                target_fps,
                ("visual",),
                max_spatial_scale,
                use_offset=use_offset,
            )
        elif backend == "decord":
            frames = container
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None

    # Return None if the frames was not decoded successfully.
    if backend in ["pyav", "torchvision"]:
        if frames is None or frames.size(0) == 0:
            return None
    elif backend == "decord":
        if frames is None:
            return None

    if backend in ["pyav", "torchvision"]:
        clip_sz = sampling_rate * num_frames / target_fps * fps
        start_idx, end_idx = get_start_end_idx(
            frames.shape[0],
            clip_sz,
            clip_idx if decode_all_video else 0,
            num_clips if decode_all_video else 1,
            use_offset=use_offset,
        )
        # Perform temporal sampling from the decoded video.
        frames = temporal_sampling(frames, start_idx, end_idx, num_frames)
    elif backend == "decord":
        if sparse:
            # Use our custom sampling method with augmentation support
            if sampling_method in ["uniform", "random", "random_window"]:
                # Use get_sampling_indices function with the specified sampling method and augmentation round
                seq = get_sampling_indices(
                    total_frames if total_frames else len(frames),
                    num_frames,
                    sampling_method,
                    clip_idx,
                    num_clips,
                    aug_round
                )
            else:
                # Fall back to original get_seq_frames function
                if total_frames:
                    seq = get_seq_frames(
                        total_frames, num_frames, clip_idx, num_clips, 
                        start_index, max_frame=len(frames)-1
                    )
                else:
                    seq = get_seq_frames(len(frames), num_frames, clip_idx, num_clips)
            frames = frames.get_batch(seq)
        else:
            clip_sz = sampling_rate * num_frames
            start_idx, end_idx = get_start_end_idx(
                len(frames),
                clip_sz,
                clip_idx,
                num_clips,
                use_offset=use_offset,
            )
            index = torch.linspace(start_idx, end_idx, num_frames)
            index = torch.clamp(index, 0, len(frames) - 1).long()
            frames = frames.get_batch(index)
       
    # Check for NaN values in frames and try to fix them
    if frames is not None:
        if backend == "decord":
            try:
                # For decord backend
                if torch.is_tensor(frames):
                    if torch.isnan(frames.float()).any():
                        print(f"Warning: NaN values detected in frames. Attempting to fix...")
                        nan_mask = torch.isnan(frames)
                        if nan_mask.any():
                            frames[nan_mask] = 0.0
                else:
                    # If frames is not a tensor, it might be a decord array
                    frames_tensor = torch.from_numpy(frames.asnumpy())
                    if torch.isnan(frames_tensor).any():
                        print(f"Warning: NaN values detected in frames. Attempting to fix...")
                        # Convert to numpy, fix NaNs, and convert back
                        frames_np = frames.asnumpy()
                        frames_np = np.nan_to_num(frames_np)
                        # This depends on how you need to return the frames
                        frames = frames_np
            except Exception as e:
                print(f"Error checking for NaNs in decord frames: {e}")
        else:
            # For PyAV and torchvision backends where frames is a tensor
            try:
                if torch.isnan(frames).any():
                    print(f"Warning: NaN values detected in frames. Attempting to fix...")
                    frames = torch.nan_to_num(frames)
            except Exception as e:
                print(f"Error checking for NaNs in frames tensor: {e}")
    return frames