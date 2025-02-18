import os
import glob
import torch
import numpy as np
import random
import clip
from PIL import Image

def extract_video_features(video_folder, preprocess, model, device):
    """
    Process a video folder: load all frames, extract features for each image,
    and return a stacked numpy array as well as the maximum frame index.
    """
    frame_paths = sorted(glob.glob(os.path.join(video_folder, "frame_*.jpg")))
    features_list = []
    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {frame_path}: {e}")
            continue
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image_input)
        
        feature = feature.cpu().numpy().reshape(-1)
        features_list.append(feature)
    if features_list:
        features_arr = np.stack(features_list)
    else:
        features_arr = np.array([])

    return features_arr, len(frame_paths) - 1

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    frames_root = '/home/goncalofigueiredo/workspace/datasets/ucfcrimeframe/frames'
    features_root = '/home/goncalofigueiredo/workspace/datasets/ucfcrimefeature/Image-Features'
    annotations_dir = '/home/goncalofigueiredo/workspace/datasets/ucfcrimefeature/Annotations'
    os.makedirs(annotations_dir, exist_ok=True)

    annotation_test = []
    annotation_train_abnormal = []
    annotation_train_normal = []
    video_entries = []

    categories = sorted(os.listdir(frames_root))
    
    class_to_id = {cat: idx for idx, cat in enumerate(categories)}

    for category in categories:
        category_path = os.path.join(frames_root, category)
        if os.path.isdir(category_path):
            video_folders = os.listdir(category_path)
            for video_folder in video_folders:
                video_path = os.path.join(category_path, video_folder)
                if os.path.isdir(video_path):
                    print(f"Processing features for {category}/{video_folder} ...")
                    features, max_frame = extract_video_features(video_path, preprocess, model, device)
                    
                    features_category_dir = os.path.join(features_root, category)
                    os.makedirs(features_category_dir, exist_ok=True)
                    features_file = os.path.join(features_category_dir, video_folder + ".npy")
                    np.save(features_file, features)
                    
                    video_entries.append((category, video_folder, max_frame, class_to_id[category]))

    random.shuffle(video_entries)
    total_videos = len(video_entries)
    test_count = int(0.2 * total_videos)
    test_entries = video_entries[:test_count]
    train_entries = video_entries[test_count:]

    for entry in test_entries:
        category, video_id, max_frame, class_id = entry
        line = f"{category}/{video_id} 0 {max_frame} {class_id}"
        annotation_test.append(line)

    for entry in train_entries:
        category, video_id, max_frame, class_id = entry
        line = f"{category}/{video_id} 0 {max_frame} {class_id}"

        if category.lower() == "normal":
            annotation_train_normal.append(line)
        else:
            annotation_train_abnormal.append(line)

    with open(os.path.join(annotations_dir, "Anomaly_Test.txt"), "w") as f:
        for line in annotation_test:
            f.write(line + "\n")
    with open(os.path.join(annotations_dir, "Anomaly_Train_Abnormal.txt"), "w") as f:
        for line in annotation_train_abnormal:
            f.write(line + "\n")
    with open(os.path.join(annotations_dir, "Anomaly_Train_Normal.txt"), "w") as f:
        for line in annotation_train_normal:
            f.write(line + "\n")

    print("Feature extraction and annotation file creation completed.")

if __name__ == '__main__':
    main()