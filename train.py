# -*- coding: utf-8 -*-
"""
Weed Segmentation with YOLOv8.
Handles data prep, training, and tuning for the Sorghum dataset.
"""

# Install dependencies
!pip install -q ultralytics pycocotools

import os
import yaml
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from google.colab import drive
from pycocotools.coco import COCO

# Mount Google Drive for persistent storage
drive.mount('/content/drive', force_remount=True)


# --- Data Preparation Functions ---

def convert_coco_to_yolo(json_path, save_dir):
    """Converts COCO segmentation annotations to YOLO .txt format."""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)
    coco = COCO(json_path)
    
    cat_ids = coco.getCatIds()
    categories = coco.loadCats(cat_ids)
    cat_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}

    for img_id in tqdm(coco.getImgIds(), desc=f"Converting {Path(json_path).stem}"):
        img_info = coco.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']
        
        label_filename = f"{Path(img_info['file_name']).stem}.txt"
        label_path = os.path.join(save_dir, label_filename)

        with open(label_path, 'w') as f:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            for ann in coco.loadAnns(ann_ids):
                if 'segmentation' in ann and ann['segmentation']:
                    yolo_class_id = cat_id_to_yolo_id[ann['category_id']]
                    for seg in ann['segmentation']:
                        norm_seg = [f"{coord / width if i % 2 == 0 else coord / height:.6f}" for i, coord in enumerate(seg)]
                        f.write(f"{yolo_class_id} {' '.join(norm_seg)}\n")


def setup_yolo_dataset(force_recreate=False):
    """
    Unzips the source dataset, converts annotations, and structures
    it for YOLOv8 training. Skips if the dataset already exists.
    """
    project_drive_path = "/content/drive/MyDrive/Sorghum_Project"
    source_zip_path = os.path.join(project_drive_path, "source_dataset/SorghumWeedDataset_Segmentation/SorghumWeedDataset_Segmentation.zip")
    processed_dataset_path = os.path.join(project_drive_path, "yolo_dataset")
    temp_unzip_path = "/content/temp_unzipped_data"

    if os.path.exists(processed_dataset_path) and not force_recreate:
        print("Processed dataset already exists. Skipping setup.")
        return os.path.join(processed_dataset_path, 'dataset.yaml')

    print("Setting up YOLO dataset from source...")
    if not os.path.exists(source_zip_path):
        raise FileNotFoundError(f"Source zip not found at {source_zip_path}")
    !unzip -q "{source_zip_path}" -d {temp_unzip_path}

    unzipped_data_root = os.path.join(temp_unzip_path, "SorghumWeedDataset_Segmentation")
    
    for split in ["Train", "Validate", "Test"]:
        yolo_split = split.lower().replace('validate', 'val')
        source_img_dir = os.path.join(unzipped_data_root, split)
        source_json = os.path.join(source_img_dir, f"{split}SorghumWeed_coco.json")
        dest_img_dir = os.path.join(processed_dataset_path, "images", yolo_split)
        dest_label_dir = os.path.join(processed_dataset_path, "labels", yolo_split)
        
        os.makedirs(dest_img_dir, exist_ok=True)
        !cp -n {source_img_dir}/*.JPG {dest_img_dir}/
        convert_coco_to_yolo(source_json, dest_label_dir)

    # Create the dataset.yaml file
    class_names = ['Sorghum', 'Grasses', 'Broad-leaf weeds']
    dataset_config = {
        'path': processed_dataset_path, 'train': 'images/train',
        'val': 'images/val', 'test': 'images/test',
        'nc': len(class_names), 'names': class_names
    }
    yaml_path = os.path.join(processed_dataset_path, 'dataset.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(dataset_config, file, default_flow_style=False, sort_keys=False)

    !rm -rf {temp_unzip_path}
    print("Dataset setup complete.")
    return yaml_path


# This only needs to be run once.
data_yaml_path = setup_yolo_dataset()


# --- Training, Evaluation, and Prediction ---

# Define project paths
project_drive_path = "/content/drive/MyDrive/Sorghum_Project"
training_runs_path = os.path.join(project_drive_path, 'training_runs')

# Baseline training run
!yolo task=segment mode=train model='yolov8s-seg.pt' data='{data_yaml_path}' epochs=50 imgsz=640 batch=8 project='{training_runs_path}' name='baseline_50_epochs'

# Training run with augmentations
!yolo task=segment mode=train model='yolov8s-seg.pt' data='{data_yaml_path}' epochs=50 imgsz=640 batch=8 project='{training_runs_path}' name='augmented_50_epochs' augment=True degrees=20 translate=0.1 scale=0.1 hsv_s=0.5 hsv_v=0.5 copy_paste=0.1

# Evaluate the augmented model (update path if needed)
best_weights_path = os.path.join(training_runs_path, 'augmented_50_epochs/weights/best.pt')
!yolo task=segment mode=val model='{best_weights_path}' data='{data_yaml_path}'

# Predict on test images
!yolo task=segment mode=predict model='{best_weights_path}' source='{os.path.join(project_drive_path, "yolo_dataset/images/test")}'


# --- Hyperparameter Tuning ---

# This is a long process to find the best training settings.
tuning_runs_path = os.path.join(project_drive_path, 'tuning_runs')

model = YOLO('yolov8s-seg.pt')

model.tune(
    data=data_yaml_path,
    epochs=50,
    iterations=100,
    imgsz=640,
    batch=8,
    project=tuning_runs_path,
    name='yolov8s_tuning_100_iters',
)