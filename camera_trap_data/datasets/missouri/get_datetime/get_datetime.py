import json
import os
import shutil
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
from tqdm import tqdm

def extract_datetime_from_exif(image_path):
    """Extract datetime from EXIF data of an image."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if exif_data is None:
            return None
            
        # Look for DateTimeOriginal first, then DateTime as fallback
        date_time_str = None
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "DateTimeOriginal":
                date_time_str = value
                break
            elif tag == "DateTime":
                date_time_str = value  # fallback
                
        if date_time_str is None:
            return None
            
        # Validate the datetime format
        try:
            dt = datetime.strptime(date_time_str, "%Y:%m:%d %H:%M:%S")
            return date_time_str
        except ValueError:
            return None
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def update_missouri_metadata_with_datetime():
    # Paths
    original_metadata_path = "/fs/ess/PAS2099/sooyoung/camera_trap_data/datasets/missouri/metadata/missouri_metadata.json"
    new_metadata_path = "/fs/ess/PAS2099/sooyoung/camera_trap_data/datasets/missouri/metadata/missouri_metadata_with_datetime.json"
    images_base_path = "/fs/ess/PAS2099/sooyoung/camera_trap_data/datasets/missouri/images"
    
    # Step 1: Copy original metadata to new location
    print("Copying original metadata file...")
    shutil.copy2(original_metadata_path, new_metadata_path)
    print(f"Copied to: {new_metadata_path}")
    
    # Step 2: Load the metadata
    print("Loading metadata...")
    with open(new_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Step 3: Process images and extract datetime - only keep images with datetime
    images_with_datetime = 0
    images_without_datetime = 0
    total_images = len(metadata.get('images', []))
    filtered_images = []  # New list to store only images with datetime
    
    print(f"Processing {total_images} images...")
    
    for image_info in tqdm(metadata['images'], desc="Processing images"):
        # Skip if no file_name field
        if 'file_name' not in image_info:
            images_without_datetime += 1
            continue
            
        # Construct full image path
        file_name = image_info['file_name']
        full_image_path = os.path.join(images_base_path, file_name)
        
        # Check if file exists
        if not os.path.exists(full_image_path):
            print(f"Warning: File not found: {full_image_path}")
            images_without_datetime += 1
            continue
            
        # Extract datetime from EXIF
        datetime_str = extract_datetime_from_exif(full_image_path)
        
        if datetime_str:
            # Add datetime to the image info and keep this image
            image_info['datetime'] = datetime_str
            filtered_images.append(image_info)
            images_with_datetime += 1
        else:
            images_without_datetime += 1
    
    # Step 4: Update metadata with filtered images
    metadata['images'] = filtered_images
    
    # Step 5: Save updated metadata
    print(f"\nSaving updated metadata...")
    with open(new_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Step 6: Print summary
    print(f"\nSummary:")
    print(f"Total images processed: {total_images}")
    print(f"Images with datetime (kept): {images_with_datetime}")
    print(f"Images without datetime (removed): {images_without_datetime}")
    print(f"Success rate: {images_with_datetime/total_images*100:.1f}%")
    print(f"Updated metadata saved to: {new_metadata_path}")

# Run the function
if __name__ == "__main__":
    update_missouri_metadata_with_datetime()