import os
import cv2
import numpy as np
import glob
import random
from pathlib import Path

DATASET_PATH = "./data/ganis_kievit_data/stimuli_jpgs"
OUTPUT_SIZE = (64, 64)
TRAIN_RATIO = 0.8

def process_ganis_dataset(root_dir):
    all_same_files = []
    all_diff_files = []

    files = glob.glob(os.path.join(root_dir, "*.jpg"))
    print(f"Found {len(files)} images. sorting...")

    for f_path in files:
        filename = Path(f_path).stem
        parts = filename.split('_')
        if len(parts) < 2: continue
        
        # 'R' tag indicates mirrored/different
        if 'R' in parts:
            all_diff_files.append(f_path)
        else:
            all_same_files.append(f_path)

    random.shuffle(all_same_files)
    random.shuffle(all_diff_files)

    # We use most "Same" pairs to teach the model rotation
    n_total_same = len(all_same_files)
    n_train = int(n_total_same * TRAIN_RATIO)
    
    train_files = all_same_files[:n_train]
    test_same_files = all_same_files[n_train:]
    
    # We must pick exactly the same number of Diff files as we have Test Same files
    n_test = len(test_same_files)
    
    test_diff_files = all_diff_files[:n_test]

    print(f"Splitting Complete:")
    print(f"  - Train (Same):      {len(train_files)}")
    print(f"  - Test (Same):       {len(test_same_files)}")
    print(f"  - Test (Diff):       {len(test_diff_files)}")
    print(f"  - Unused (Diff):     {len(all_diff_files) - len(test_diff_files)} (Discarded to maintain balance)")

    def process_files(file_list, label_name):
        processed_data = []
        for f_path in file_list:
            filename = Path(f_path).stem
            parts = filename.split('_')
            angle = int(parts[1])

            img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            h, w = img.shape
            img_left = img[:, :w // 2]
            img_right = img[:, w // 2:]

            x0 = cv2.resize(img_left, OUTPUT_SIZE)
            x1 = cv2.resize(img_right, OUTPUT_SIZE)
            
            x0 = (x0.astype(np.float32) / 127.5) - 1.0
            x1 = (x1.astype(np.float32) / 127.5) - 1.0

            processed_data.append({
                'x0': np.expand_dims(x0, axis=0),
                'x1': np.expand_dims(x1, axis=0),
                'angle': angle,
                'name': filename,
                'label': label_name
            })
        return processed_data

    train_data = process_files(train_files, 'same')
    test_same_data = process_files(test_same_files, 'same')
    test_diff_data = process_files(test_diff_files, 'diff')

    full_test_set = test_same_data + test_diff_data
    random.shuffle(full_test_set)

    return train_data, full_test_set

if __name__ == "__main__":
    train, test = process_ganis_dataset(DATASET_PATH)
    
    np.save("train_pairs.npy", train)
    np.save("test_balanced.npy", test)
    print("Saved balanced .npy files.")