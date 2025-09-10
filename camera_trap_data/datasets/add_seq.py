# This script processes camera trap metadata to assign sequence IDs and frame numbers to images based on their timestamps.

# Import modules
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import shortuuid
import argparse
import wandb
import json
import os

# Parse command line arguments
# parser = argparse.ArgumentParser(description = "Process node ID")
# parser.add_argument('--node_id', type = int, required = True, help = 'Unique node identifier')
# args = parser.parse_args()
# node_id = 0

# Initialize run in wandb
# wandb.init(project = "location-processing-2", config = {"node_id": node_id})

# Load the metadata from the JSON file
path = "/fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/dlc/metadata/dlc_metadata.json"
with open(path, "r") as file:
    metadata = json.load(file)

# Create a dictionary to hold the node data
# Directly copy info + categories
# Only these camera's images + annotations
node_data = {
    "info": metadata["info"],
    "categories": metadata["categories"],
    "images": [],
    "annotations": []
}

# Assign locations to each node
locations = {image["location"] for image in metadata["images"]}
locations = sorted(locations)

# Calculate the number of images for the current node
# processed_count = 0
# for image in tqdm(metadata["images"], desc = f"Node {node_id} processing images"):
    # if image["location"] in locations:
        # processed_count += 1

# Log the number of locations and images processed for the current node
# wandb.log({
    # "num_locations": len(locations),
    # "images_processed": processed_count,
# })

# Iterate over each location for the current node
for location in locations:
    # Filter images for the current location
    filtered_images = [image for image in metadata["images"] if image["location"] == location]

    # Ensure the filtered images have a datetime field
    if all("datetime" in image and image["datetime"] is not None for image in filtered_images):
        # Sort the filtered images by datetime
        sorted_images = sorted(filtered_images, key=lambda x: datetime.strptime(x["datetime"], "%Y:%m:%d %H:%M:%S"))

        # Assign sequence IDs to images
        for image_1, image_2 in combinations(sorted_images, 2):
            if "seq_id" not in image_1:
                image_1["seq_id"] = shortuuid.uuid()   

            if "seq_id" not in image_2:
                # Find the datetime strings for the two images
                date_1 = image_1["datetime"]
                date_2 = image_2["datetime"]
                
                # Convert the datetime strings to datetime objects
                date_1 = datetime.strptime(date_1, "%Y:%m:%d %H:%M:%S")
                date_2 = datetime.strptime(date_2, "%Y:%m:%d %H:%M:%S")

                # Calculate the difference in seconds
                diff = abs((date_2 - date_1).total_seconds())

                # Assign the same seq_id if the difference is less than or equal to 1 second
                if diff <= 1:
                    image_2["seq_id"] = image_1["seq_id"]

        # If the last image does not have a seq_id, assign a new one
        if "seq_id" not in sorted_images[-1]:
                sorted_images[-1]["seq_id"] = shortuuid.uuid()

        # Assign sequence numbers and frame numbers
        seq_ids = {image["seq_id"] for image in sorted_images}
        for seq_id in seq_ids:
            seq = [image for image in sorted_images if image["seq_id"] == seq_id]
            seq_num_frames = len(seq)
            for i in range(seq_num_frames):
                seq[i]["seq_num_frames"] = seq_num_frames
                seq[i]["frame_num"] = i
        
        # Add the sorted images to the node data
        node_data["images"].extend(sorted_images)

        # Add annotations for the sorted images
        sorted_image_ids = {image["id"] for image in sorted_images}
        sorted_annotations = [annotation for annotation in metadata["annotations"] if annotation["image_id"] in sorted_image_ids]
        node_data["annotations"].extend(sorted_annotations)

# Save the updated metadata back to the JSON file
output_path = f"/fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/dlc/metadata/dlc_metadata_with_seq.json"
with open(output_path, "w") as write_file:
    json.dump(node_data, write_file, indent = 4)