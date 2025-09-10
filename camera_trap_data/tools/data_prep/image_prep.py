import os
import json
import glob
import cv2
from tqdm import tqdm
from .utils.loader_utils import get_taxonomy_loader
from .utils.datetime_utils import normalize_datetime

def crop_and_save_images_once(dataset, output_path):
    """
    Crop and save images only once per camera, regardless of day intervals.
    This should be called only once per camera to avoid redundant image processing.

    Args:
        dataset (Dataset): The dataset object with prepared checkpoints.
        output_path (str): The root directory to save cropped images.
    """
    print("\n======================================== Cropping Images (One-Time) =========================================\n")

    # Iterate through each dataset
    for dataset_name, cameras in dataset.metadata.items():
        dataset_dir = os.path.join(output_path, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        total_cameras = len(cameras)  # Total number of cameras

        # Iterate through each camera
        for camera_idx, (camera_name, camera_data) in enumerate(cameras.items(), start=1):
            camera_dir = os.path.join(dataset_dir, camera_name)
            images_dir = os.path.join(camera_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Check if images are already cropped
            if os.path.exists(os.path.join(camera_dir, ".images_cropped")):
                print(f"📸 Images already cropped for {camera_name}, skipping...")
                continue

            print(f"📸 Cropping images for {camera_name} ({camera_idx}/{total_cameras})")

            # Collect all unique images across all checkpoints
            all_images = {}  # image_path -> image_data
            
            for ckp_id, ckp_data in camera_data["ckp"].items():
                # Collect from both train and val
                for img in ckp_data["train"] + ckp_data["val"]:
                    if img["image_path"] not in all_images:
                        all_images[img["image_path"]] = img

            # Process all unique images
            for img_path, img_data in tqdm(
                all_images.items(),
                desc=f"Cropping images for {camera_name}",
                leave=False,
            ):
                process_and_crop_image(img_data, images_dir)

            # Create a marker file to indicate images are cropped
            with open(os.path.join(camera_dir, ".images_cropped"), "w") as f:
                f.write("Images cropped successfully")

            print(f"✅ Completed cropping for {camera_name}")

def prepare_day_specific_jsons(dataset, output_path, checkpoint_days):
    """
    Prepare JSON files for a specific day interval without re-cropping images.

    Args:
        dataset (Dataset): The dataset object with prepared checkpoints.
        output_path (str): The root directory to save JSON metadata.
        checkpoint_days (int): Number of days for the checkpoint interval.
    """
    print(f"\n======================================== Preparing JSONs ({checkpoint_days} days) =========================================\n")

    # Iterate through each dataset
    for dataset_name, cameras in dataset.metadata.items():
        dataset_dir = os.path.join(output_path, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        total_cameras = len(cameras)  # Total number of cameras

        # Iterate through each camera
        for camera_idx, (camera_name, camera_data) in enumerate(cameras.items(), start=1):
            camera_dir = os.path.join(dataset_dir, camera_name)
            images_dir = os.path.join(camera_dir, "images")
            
            # Create day-specific directory
            day_dir = os.path.join(camera_dir, str(checkpoint_days))
            os.makedirs(day_dir, exist_ok=True)

            # Initialize JSON metadata for train and test
            train_metadata = {}
            test_metadata = {}

            # Process each checkpoint with tqdm
            for ckp_id, ckp_data in tqdm(
                camera_data["ckp"].items(),
                desc=f"({camera_idx}/{total_cameras}) Processing JSONs for {camera_name} ({checkpoint_days} days)",
                leave=False,
            ):
                # Process train images
                train_metadata[f"ckp_{ckp_id}"] = process_ckp_metadata(
                    ckp_data["train"], images_dir, dataset_name, camera_name
                )

                # Process test images
                test_metadata[f"ckp_{ckp_id}"] = process_ckp_metadata(
                    ckp_data["val"], images_dir, dataset_name, camera_name
                )

            # Save train and test JSON metadata in day-specific directory
            train_all_json_path = os.path.join(day_dir, "train-all.json")
            train_json_path = os.path.join(day_dir, "train.json")
            test_json_path = os.path.join(day_dir, "test.json")

            with open(train_json_path, "w") as train_file:
                json.dump(train_metadata, train_file, indent=2)

            with open(test_json_path, "w") as test_file:
                json.dump(test_metadata, test_file, indent=2)

            # --------- Create train-all.json with merged ckps ---------
            
            # Load train.json
            with open(train_json_path, "r") as f:
                train_data = json.load(f)

            # Collect all ckp_* keys from train_data
            merged_ckp_entries = []
            keys_to_delete = []

            for key in train_data:
                merged_ckp_entries.extend(train_data[key])
                keys_to_delete.append(key)

            # Delete original ckp_* keys
            for key in keys_to_delete:
                del train_data[key]

            # Add merged entries under ckp_-1
            train_data["ckp_-1"] = merged_ckp_entries

            # Save the result
            with open(train_all_json_path, "w") as f:
                json.dump(train_data, f, indent=2)

            print(f"📄 JSON files saved for {camera_name} ({checkpoint_days} days)")
            print(f"   Train JSON: {train_json_path}")
            print(f"   Test JSON: {test_json_path}")
            print(f"   Train-all JSON: {train_all_json_path}")

def process_and_crop_image(img_data, images_dir):
    """
    Process a single image: crop it and save the cropped version.

    Args:
        img_data (dict): Image data dictionary.
        images_dir (str): Directory to save cropped images.
    """
    # Load the image
    image_path = img_data["image_path"]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Get bounding box and crop the image
    bboxes = img_data.get("bbox", None)
    if bboxes:
        # Select the bbox with the highest confidence and size
        selected_bbox = max(
            bboxes,
            key=lambda b: (b["conf"], b["bbox"][2] * b["bbox"][3])  # Sort by confidence, then by area
        )
        x, y, w, h = selected_bbox["bbox"]

        # Convert normalized bbox to absolute pixel values
        img_h, img_w = image.shape[:2]
        x = int(x * img_w)
        y = int(y * img_h)
        w = int(w * img_w)
        h = int(h * img_h)

        # Scale the bounding box by 1.5x
        cx, cy = x + w / 2, y + h / 2  # Center of the bounding box
        new_w, new_h = w * 1.5, h * 1.5
        new_x = max(int(cx - new_w / 2), 0)
        new_y = max(int(cy - new_h / 2), 0)
        new_x2 = min(int(cx + new_w / 2), img_w)
        new_y2 = min(int(cy + new_h / 2), img_h)
        cropped_image = image[new_y:new_y2, new_x:new_x2]

        # Save the cropped image
        cropped_image_name = os.path.basename(image_path)
        cropped_image_path = os.path.join(images_dir, cropped_image_name)
        cv2.imwrite(cropped_image_path, cropped_image)
    else:
        # Skip the image if no bounding box is provided
        print(f"Warning: No bounding box found for image {image_path}. Skipping this image.")

def process_ckp_metadata(images, images_dir, dataset_name, camera_name):
    """
    Process images in a checkpoint and generate metadata (without cropping).

    Args:
        images (list): List of image data dictionaries.
        images_dir (str): Directory where cropped images are saved.
        dataset_name (str): Name of the dataset.
        camera_name (str): Name of the camera.

    Returns:
        list: Metadata for the processed images.
    """
    metadata = []
    missing_images_count = 0
    
    # Get taxonomy loader for scientific/common name lookups
    taxonomy_loader = get_taxonomy_loader()

    for img in images:
        # Get bounding box info
        bboxes = img.get("bbox", None)
        if bboxes:
            # Select the bbox with the highest confidence and size
            selected_bbox = max(
                bboxes,
                key=lambda b: (b["conf"], b["bbox"][2] * b["bbox"][3])  # Sort by confidence, then by area
            )

            # Use the cropped image path
            cropped_image_name = os.path.basename(img["image_path"])
            cropped_image_path = os.path.join(images_dir, cropped_image_name)
            
            # Check if the cropped image actually exists
            if not os.path.exists(cropped_image_path):
                missing_images_count += 1
                if missing_images_count <= 5:  # Only print first 5 to avoid spam
                    print(f"Warning: Cropped image not found: {cropped_image_path}. Skipping from JSON.")
                elif missing_images_count == 6:
                    print(f"... (more missing cropped images, suppressing further warnings for {camera_name})")
                continue  # Skip this image from JSON metadata
            
            # Get class information
            class_info = img["class"][0] if img["class"] else {}
            class_name = class_info.get("class_name", "")
            class_id = class_info.get("class_id", None)
            
            # Get taxonomy information
            taxonomy_info = taxonomy_loader.get_taxonomy_info(dataset_name, class_name)
            scientific_name = ""
            common_name = ""
            
            if taxonomy_info:
                scientific_name, common_name = taxonomy_info

            # Create metadata entry with taxonomy information
            try:
                normalized_datetime = normalize_datetime(img["datetime"])
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not normalize datetime '{img['datetime']}' for image {img['image_path']}. Using original format. Error: {e}")
                normalized_datetime = img["datetime"]
            
            metadata_entry = {
                "image_path": cropped_image_path,
                "class_id": class_id,
                "query": class_name,  # Changed from class_name to query
                "scientific": scientific_name,
                "common": common_name,
                "datetime": normalized_datetime,  # Normalize datetime format
                "conf": selected_bbox["conf"],
            }
            
            # Add sequence ID if available
            if "seq" in img:
                metadata_entry["seq_id"] = img["seq"]["seq_id"]
            
            metadata.append(metadata_entry)
        else:
            # Skip the image if no bounding box is provided
            print(f"Warning: No bounding box found for image {img['image_path']}. Skipping this image.")

    if missing_images_count > 0:
        print(f"📸 Filtered out {missing_images_count} missing cropped images from {camera_name} JSON metadata")

    return metadata

def crop_and_save_images(dataset, output_path, save_images=True, checkpoint_days=30):
    """
    Optimized function that either crops images (one-time) or prepares JSONs for specific day intervals.
    
    Args:
        dataset (Dataset): The dataset object with prepared checkpoints.
        output_path (str): The root directory to save cropped images and JSON metadata.
        save_images (bool): If True, will crop images if not already done.
        checkpoint_days (int): Number of days for the checkpoint interval.
    """
    if save_images:
        # Crop images only once per camera
        crop_and_save_images_once(dataset, output_path)
    
    # Always prepare day-specific JSON files
    prepare_day_specific_jsons(dataset, output_path, checkpoint_days)
