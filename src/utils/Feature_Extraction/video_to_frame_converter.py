import os
import math
import cv2
import glob

def extract_frames(video_path, out_dir, target_fps=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / original_fps

    num_frames = math.floor(duration * target_fps)
    for frame_idx in range(num_frames):
        t_sec = frame_idx / target_fps
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame image
        frame_filename = os.path.join(out_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
    
    cap.release()
    print(f"Extracted {num_frames} frames from {video_path}")

def process_videos(source_root, dest_root, video_extensions=('mp4', 'avi', 'mov')):

    for category in os.listdir(source_root):
        category_path = os.path.join(source_root, category)
        if os.path.isdir(category_path):
            for ext in video_extensions:
                for video_path in glob.glob(os.path.join(category_path, f'*.{ext}')):
                    # Create an output folder based on class and video file name
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    out_dir = os.path.join(dest_root, category, video_name)
                    os.makedirs(out_dir, exist_ok=True)
                    print(f"Processing {video_path}...")
                    extract_frames(video_path, out_dir)

if __name__ == '__main__':

    source_root = '/home/goncalofigueiredo/workspace/datasets/ucfcrimevideo'
    dest_root = '/home/goncalofigueiredo/workspace/datasets/ucfcrimeframe/frames'
    
    process_videos(source_root, dest_root)