import json
from pycococt.cococt import COCOCT
import os
import yaml
from contextlib import redirect_stdout
from tqdm import tqdm

class NaDataset:
    def __init__(self, dataset_config):
        """
        NA-specific dataset class.

        Args:
            dataset_config (dict): Configuration for the NA dataset.
        """
        self.dataset_config = dataset_config
        self.ds = None  # COCOCT object to store dataset
        self.image_root = "images/nacti-unzipped"  # Root directory for images
        self.bbox_data = None  # To store loaded bbox data

    def load_metadata(self):
        """
        Load and organize NA-specific metadata based on the configuration.

        Returns:
            dict: Organized metadata structure.
        """
        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)
        self.image_root = os.path.join(current_dir, "images/nacti-unzipped")

        # Construct absolute paths for metadata and bbox files
        metadata_path = os.path.join(current_dir, "metadata", "na_metadata.json")
        bbox_path = os.path.join(current_dir, "metadata", "na_bbox.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not os.path.exists(bbox_path):
            raise FileNotFoundError(f"Bbox file not found: {bbox_path}")

        # Suppress pycococt messages
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
            self.ds = COCOCT(metadata_path)

        self._load_bbox_data(bbox_path)

        na_config = self.dataset_config
        # all_datasets = safari_config.get("all_datasets", False)
        # selected_datasets = safari_config.get("datasets", []) if not all_datasets else list(safari_config.keys())

        # if all_datasets:
            # selected_datasets = [
                # key for key in safari_config.keys()
                # if key not in ["all_datasets", "datasets", "filtering_config"]
            # ]

        organized_metadata = {}

        # Iterate through the selected datasets with tqdm
        # for dataset_name in tqdm(selected_datasets, desc="Processing datasets", leave=False):
        # dataset_config = safari_config.get(dataset_name, {})
        dataset_name = "na"

        all_camera = na_config.get("all_camera", False)
        cameras = na_config.get("cameras", [])

        # Merge dataset_name with cameras
        cameras = [f"{dataset_name}_{camera}" for camera in cameras]

        # Filter data based on cameras
        if all_camera:
            organized_metadata.update(self._get_camera_data(dataset_name))
        else:
            organized_metadata.update(self._get_camera_data(dataset_name, cameras))

        # Add bbox data to the organized metadata
        self._get_bbox_data(organized_metadata)
        self.summarize_data(organized_metadata)

        return organized_metadata

    def summarize_data(self, organized_metadata):
        """
        Summarize the retrieved data for the NA dataset.

        Args:
            organized_metadata (dict): The organized metadata structure.

        Returns:
            dict: A summary of the dataset and its associated cameras.
        """
        summary = {}
        for dataset_name, dataset_metadata in organized_metadata.items():
            # Extract all unique cameras from the metadata
            cameras = set(entry["location"] for entry in dataset_metadata["data"])
            cameras = sorted(cameras)  # Sort cameras for consistent output

            # Prepare the summary for the dataset
            if len(cameras) > 5:
                summary[dataset_name] = {
                    "cameras": cameras[:5],  # First 5 cameras
                    "remaining": len(cameras) - 5,  # Remaining camera count
                }
            else:
                summary[dataset_name] = {
                    "cameras": cameras,  # All cameras if <= 5
                    "remaining": 0,  # No remaining cameras
                }

        # Print the summary
        print("na:")
        for dataset_name, dataset_info in summary.items():
            # print(f"  {dataset_name}:")
            if dataset_info["remaining"] > 0:
                camera_list = ", ".join(dataset_info["cameras"])
                print(f"    [{camera_list}, .... + {dataset_info['remaining']} more cameras]")
            else:
                camera_list = ", ".join(dataset_info["cameras"])
                print(f"    [{camera_list}]")
        print()
        return summary

    def _load_bbox_data(self, bbox_path):
        """
        Load the bounding box data from the bbox.json file.

        Args:
            bbox_path (str): Path to the bbox.json file.
        """
        if not os.path.exists(bbox_path):
            raise FileNotFoundError(f"BBox file not found: {bbox_path}")

        with open(bbox_path, "r") as f:
            self.bbox_data = json.load(f)

    def _get_bbox_data(self, organized_metadata):
        """
        Match and fill bbox data into the organized metadata.

        Args:
            organized_metadata (dict): Metadata for all datasets.
        """
        # Create a lookup dictionary for bbox data
        bbox_lookup = {entry["file"]: entry["detections"] for entry in self.bbox_data["images"]}

        # Iterate through the organized metadata and fill bbox data
        for dataset_name, dataset_metadata in organized_metadata.items():
            for entry in dataset_metadata["data"]:
                image_file = entry["image_path"].split("images/nacti-unzipped/")[-1]  # Use image_id to match with bbox file
                if image_file in bbox_lookup:
                    entry["bbox"] = bbox_lookup[image_file]
                else:
                    entry["bbox"] = []  # No detections for this image

    def _get_camera_data(self, dataset_name, cameras=None):
        """
        Retrieve data entries for cameras in NA.

        Args:
            dataset_name (str): Name of the dataset.
            cameras (list or None): List of camera names. If None, include all cameras.

        Returns:
            dict: Metadata organized by dataset name.
        """
        data = [
            {
                "image_id": img_id,
                "image_path": os.path.join(self.image_root, self.ds.imgs[img_id]["file_name"]),
                "class": [
                    {
                        "class_id": ann["category_id"],
                        "class_name": self.ds.cats[ann["category_id"]]["name"]
                    }
                    for ann in self.ds.imgToAnns[img_id]
                ] if img_id in self.ds.imgToAnns else [{"class_id": "unknown", "class_name": "unknown"}],
                "bbox": [],  # Placeholder for bbox data
                "location": self._format_location(self.ds.imgs[img_id]["location"]),
                "datetime": self.ds.imgs[img_id].get("datetime"),
                "seq": {
                    "seq_id": self.ds.imgs[img_id].get("seq_id", None),
                    "seq_num_frames": self.ds.imgs[img_id].get("seq_num_frames", None),
                    "frame_num": self.ds.imgs[img_id].get("frame_num", None),
                },
            }
            for img_id in self.ds.imgs
            if (cameras is None or self._format_location(self.ds.imgs[img_id]["location"]) in cameras)  # Match cameras
            and (self.ds.imgs[img_id].get("datetime")) # Only retrieve images with datetime
        ]

        return {
            dataset_name: {
                "dataset_name": f"{dataset_name}",
                "filtering_config": self.dataset_config.get("filtering_config", {}),
                "data": data,
            }
        }

    def _format_location(self, location):
        """
        Format the location to add NA at the beginning.

        Args:
            location (str): Original location string (e.g., "CA-30").

        Returns:
            str: Formatted location string (e.g., "NA_CA-30").
        """
        # parts = location.split("_")
        # if len(parts) >= 2:
            # return f"{parts[0]}_{parts[-1]}"  # Keep only the first and last parts
        
        dataset_name = "na"
        return f"{dataset_name}_{location}"

    # def _get_dataset_name(self, location):
        # """
        # Extract the dataset name from the location.

        # Args:
            # location (str): Original location string (e.g., "APN_S2_13U").

        # Returns:
            # str: Dataset name (e.g., "APN").
        # """
        # return location.split("_")[0]  # The first part of the location is the dataset name