import os
import csv

# Define paths and labels
base_path = "data/laryngeal_dataset_balanced/dataset"  # Update this to your actual base path if different
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