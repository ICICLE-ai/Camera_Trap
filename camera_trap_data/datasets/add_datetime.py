from datetime import datetime
import json

# Load your JSON metadata (if it's in a file)
with open("/fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/channel/metadata/channel_metadata.json", "r") as file:
    data = json.load(file)

for image in data["images"]:
    path = image.get("original_relative_path", "")
    try:
        # Extract the date string from the path
        parts = path.split("/")
        if len(parts) >= 4:
            date_str = parts[3]  # e.g., "2011-09-13"
            # Validate or reformat if needed
            datetime.strptime(date_str, "%Y-%m-%d")  # will raise if invalid
            image["datetime"] = date_str
        else:
            print(f"Unexpected path format: {path}")
            image["datetime"] = None
    except Exception as e:
        print(f"Error parsing datetime from path: {path}, error: {e}")
        image["datetime"] = None

# Save the updated data
with open("/fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/channel/metadata/channel_metadata_with_datetime.json", "w") as out_file:
    json.dump(data, out_file, indent=4)