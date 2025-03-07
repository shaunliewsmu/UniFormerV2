import os
import decord
import glob

problem_files = []
for video_path in glob.glob("data/laryngeal_dataset_balanced/dataset/**/*.mp4", recursive=True):
    try:
        # Try to load with decord
        vr = decord.VideoReader(video_path)
        # Test access to a few frames
        vr.get_batch([0, 1, 2])
    except Exception as e:
        print(f"Problem with {video_path}: {e}")
        problem_files.append(video_path)

print(f"Found {len(problem_files)} problematic files")
print(problem_files)