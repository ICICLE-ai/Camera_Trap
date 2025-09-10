import random
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
try:
    # Prefer centralized datetime utils
    from .utils.datetime_utils import normalize_datetime
except Exception:
    normalize_datetime = None

def parse_datetime(datetime_str, datetime_formats=None):
    """
    Parse datetime string using multiple format attempts.
    
    Args:
        datetime_str (str): The datetime string to parse.
        datetime_formats (list): List of datetime format strings to try. If None, uses default.
        
    Returns:
        datetime: Parsed datetime object.
        
    Raises:
        ValueError: If no format matches the datetime string.
    """
    # Default datetime formats if none provided
    if datetime_formats is None:
        datetime_formats = [
            "%Y:%m:%d %H:%M:%S",  # Original format: "2016:01:08 10:35:00"
            "%Y-%m-%d %H:%M:%S",  # New format: "2016-01-08 10:35:00"
            "%Y/%m/%d %H:%M:%S",  # Alternative format: "2016/01/08 10:35:00"
            "%d/%m/%Y %H:%M",     # Format: "19/09/2015 10:12"
            "%d/%m/%Y %H:%M:%S",  # Format: "19/09/2015 10:12:30"
            "%m/%d/%Y %H:%M",     # Format: "4/14/2015 9:19" (American format)
            "%m/%d/%Y %H:%M:%S",  # Format: "4/14/2015 9:19:30" (American format with seconds)
            "%Y-%m-%d",           # Date only: "2016-01-08"
            "%Y:%m:%d",           # Date only with colons: "2016:01:08"
            "%Y/%m/%d",           # Date only with slashes: "2016/01/08"
            "%d/%m/%Y",           # Date only: "19/09/2015"
            "%m/%d/%Y",           # Date only: "4/14/2015"
        ]
    
    # First try centralized normalizer if available
    if normalize_datetime is not None:
        try:
            normalized = normalize_datetime(datetime_str)
            return datetime.strptime(normalized, "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass

    for fmt in datetime_formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    
    # If no format works, raise an error with helpful information
    raise ValueError(f"Unable to parse datetime '{datetime_str}'. Supported formats: {datetime_formats}")

def prepare_checkpoints(dataset, datetime_formats=None, checkpoint_days=30):    
    """
    Process the dataset to split each camera's data into checkpoints (ckp) and further divide into train, val, and rare.

    Args:
        dataset (Dataset): The dataset object with filtered and organized metadata.
        datetime_formats (list): List of datetime format strings to try when parsing dates.
        checkpoint_days (int): Number of days for each checkpoint interval.
    """
    print(f"\n======================================== Preparing Ckp (Day Interval: {checkpoint_days}) ============================================\n")    # Add tqdm for progress tracking
    dataset_progress = tqdm(dataset.metadata.items(), desc="Processing datasets", leave=False)

    for dataset_name, cameras in dataset_progress:
        for camera_name, camera_data in cameras.items():            # Sort data by datetime
            data = sorted(camera_data["data"], key=lambda x: parse_datetime(x["datetime"], datetime_formats))
            ckp = defaultdict(lambda: {"train": [], "val": [], "rare": []})
            ckp_number = 1
            current_ckp = []
            start_date = None

            # Split data into checkpoints
            for entry in data:
                entry_date = parse_datetime(entry["datetime"], datetime_formats)
                if start_date is None:
                    start_date = entry_date

                # Check if specified days have passed or if the current checkpoint has fewer than 200 images
                if (entry_date - start_date).days > checkpoint_days and len(current_ckp) >= 200:
                    # Finalize the current checkpoint
                    ckp[ckp_number]["data"] = current_ckp
                    ckp_number += 1
                    current_ckp = []
                    start_date = entry_date

                current_ckp.append(entry)

            # Add the last checkpoint if it has data
            if current_ckp:
                ckp[ckp_number]["data"] = current_ckp

            # Merge checkpoints with fewer than 200 images
            merged_ckp = defaultdict(lambda: {"data": []})
            ckp_number = 1
            for ckp_id, ckp_data in ckp.items():
                if len(merged_ckp[ckp_number]["data"]) + len(ckp_data["data"]) < 200:
                    merged_ckp[ckp_number]["data"].extend(ckp_data["data"])
                else:
                    if merged_ckp[ckp_number]["data"]:
                        ckp_number += 1
                    merged_ckp[ckp_number]["data"].extend(ckp_data["data"])

            # Merge the very last checkpoint with the second-to-last if it has fewer than 200 images
            if len(merged_ckp) > 1:  # Ensure there are at least two checkpoints
                last_ckp_id = max(merged_ckp.keys())  # Get the last checkpoint ID
                second_last_ckp_id = last_ckp_id - 1  # Get the second-to-last checkpoint ID

                # Check if the last checkpoint has fewer than 200 images
                if len(merged_ckp[last_ckp_id]["data"]) < 200:
                    # Merge the last checkpoint into the second-to-last checkpoint
                    merged_ckp[second_last_ckp_id]["data"].extend(merged_ckp[last_ckp_id]["data"])
                    del merged_ckp[last_ckp_id]  # Remove the last checkpoint after merging

            # Analyze each checkpoint and split into train, val, and rare
            for ckp_id, ckp_data in merged_ckp.items():
                class_counts = defaultdict(int)
                for entry in ckp_data["data"]:
                    class_counts[entry["class"][0]["class_id"]] += 1

                # Separate rare classes
                rare_classes = {cls for cls, count in class_counts.items() if count < 10}
                rare_data = []
                valid_data = []

                for entry in ckp_data["data"]:
                    if entry["class"][0]["class_id"] in rare_classes:
                        rare_data.append(entry)
                    else:
                        valid_data.append(entry)

                merged_ckp[ckp_id]["rare"] = rare_data

                # Split valid data into train and val
                class_groups = defaultdict(list)
                for entry in valid_data:
                    class_groups[entry["class"][0]["class_id"]].append(entry)

                train_data = []
                val_data = []

                for cls, entries in class_groups.items():
                    # Check if sequence information is available
                    if "seq" in entries[0] and entries[0]["seq"]["seq_id"] is not None:
                        # Group by sequence
                        sequence_groups = defaultdict(list)
                        for entry in entries:
                            sequence_groups[entry["seq"]["seq_id"]].append(entry)

                        # Randomly shuffle sequences
                        sequences = list(sequence_groups.values())
                        random.shuffle(sequences)

                        # Fill val first with at least 10 images
                        val_count = 0
                        for seq in sequences:
                            if val_count >= 10:
                                break
                            val_data.extend(seq)
                            val_count += len(seq)

                        # Add remaining sequences to train
                        for seq in sequences:
                            if all(entry in val_data for entry in seq):
                                continue
                            train_data.extend(seq)
                    else:
                        # Randomly shuffle entries
                        random.shuffle(entries)

                        # Split into train and val
                        val_data.extend(entries[:10])  # Store first 10 images in val
                        train_data.extend(entries[10:])

                merged_ckp[ckp_id]["train"] = train_data
                merged_ckp[ckp_id]["val"] = val_data

                # Remove the redundant "data" field
                del merged_ckp[ckp_id]["data"]

            # Update the metadata with the checkpoint information
            camera_data["ckp"] = merged_ckp

    # Close the tqdm progress bar
    dataset_progress.close()