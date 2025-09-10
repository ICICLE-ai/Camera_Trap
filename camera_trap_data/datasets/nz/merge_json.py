# This script merges multiple JSON files containing metadata for camera trap images.

# Import modules
from glob import glob
import json
import os

# Define the path to the directory containing the JSON files
path = "/fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/nz/metadata"
json_files = glob(os.path.join(path, "trail_camera_images_of_new_zealand_animals_1.00_with_seq_*.json"))

# Create a dictionary to hold the merged data
merged = {
    "info": None,
    "categories": None,
    "images": [],
    "annotations": []
}

# Iterate over each JSON file and merge the data
for file_path in json_files:
    with open(file_path, "r") as file:
        data = json.load(file)

        # Ensure the info and categories are only set once
        if merged["info"] is None:
            merged["info"] = data["info"]
        if merged["categories"] is None:
            merged["categories"] = data["categories"]

        merged["images"].extend(data["images"])
        merged["annotations"].extend(data["annotations"])

# Write the merged data to a new JSON file
output_path = "/fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/nz/metadata/trail_camera_images_of_new_zealand_animals_1.00_with_seq.json"
with open(output_path, "w") as write_file:
    json.dump(merged, write_file, indent = 4)