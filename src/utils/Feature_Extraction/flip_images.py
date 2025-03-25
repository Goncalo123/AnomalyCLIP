import cv2
import numpy as np
from PIL import Image
import os 
import glob

def get_frames():
    frames = []
    for filename in glob.iglob('D:/trafficwsal/Frames/*/*/*.jpg', recursive=True):
        frames.append(filename)
    return frames

def flip_image(image):
    return cv2.flip(image, 1)

def main():
    input_root = 'D:/trafficwsal/Frames'
    output_root = 'D:/flippedtraffic/Frames'
    
    frames = get_frames()
    for frame_path in frames:
        if os.path.commonpath([os.path.join(input_root, 'Normal'), frame_path]) == os.path.join(input_root, 'Normal'):
            continue

        image = cv2.imread(frame_path)
        if image is None:
            print(f"Failed to load {frame_path}")
            continue

        flipped = flip_image(image)
        rel_path = os.path.relpath(frame_path, input_root)
        
        out_path = os.path.join(output_root, rel_path)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, flipped)
        print(f"Saved flipped image: {out_path}")

if __name__ == '__main__':
    main()