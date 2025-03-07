import cv2
import os
import numpy as np

def visualize_sampled_frames(frames, video_path, output_dir, sampling_method):
    """
    Visualize the sampled frames from a video.
    
    Args:
        frames (tensor or ndarray): Sampled frames
        video_path (str): Path to the original video
        output_dir (str): Directory to save visualizations
        sampling_method (str): Sampling method used
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename without extension
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Ensure frames are in (N, H, W, C) format
    # Check if frames are in (N, C, H, W) format and transpose if needed
    if isinstance(frames, np.ndarray):
        if frames.shape[1] == 3 and frames.shape[3] != 3:
            # Frames are in (N, C, H, W) format, transpose to (N, H, W, C)
            frames = frames.transpose(0, 2, 3, 1)
    
    # Get frame dimensions
    num_frames = frames.shape[0]
    frame_height = frames.shape[1]
    frame_width = frames.shape[2]
    
    # Create a grid of frames
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    
    # Calculate grid dimensions
    grid_height = grid_size * frame_height
    grid_width = grid_size * frame_width
    
    # Create an empty grid
    grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Fill the grid with frames
    for i in range(num_frames):
        row = i // grid_size
        col = i % grid_size
        
        y1 = row * frame_height
        y2 = y1 + frame_height
        x1 = col * frame_width
        x2 = x1 + frame_width
        
        frame = frames[i]
        
        # Ensure frame is in the right format (H, W, C) with RGB channels
        if frame.shape[0] == 3 and len(frame.shape) == 3:
            # Frame is in (C, H, W) format
            frame = frame.transpose(1, 2, 0)
        
        # Ensure frame is uint8 for OpenCV
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        grid_img[y1:y2, x1:x2] = frame
    
    # Add sampling method text
    cv2.putText(
        grid_img, 
        f"Sampling: {sampling_method}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )
    
    # Save the grid image
    output_path = os.path.join(output_dir, f"{video_name}_{sampling_method}_frames.jpg")
    cv2.imwrite(output_path, grid_img)
    
    return output_path