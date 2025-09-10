from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# Load the metadata from the JSON file
path = "/fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/na/metadata/na_metadata.json"
with open(path, "r") as file:
    metadata = json.load(file)

# Filter the metadata to include only images from the specified location
location = "lebec_CA-14"
filtered_images = [image for image in metadata["images"] if image["location"] == location]

# Sort the filtered images by datetime
sorted_images = sorted(filtered_images, key=lambda x: datetime.strptime(x["datetime"], "%Y:%m:%d %H:%M:%S"))

# Extract the datetimes from the sorted images
datetimes = [image["datetime"] for image in sorted_images]

# Determine the number of images taken within 60 seconds of each other
counts = defaultdict(int)
for date_1, date_2 in combinations(datetimes, 2):
    # Convert the datetime strings to datetime objects
    date_1 = datetime.strptime(date_1, "%Y:%m:%d %H:%M:%S")
    date_2 = datetime.strptime(date_2, "%Y:%m:%d %H:%M:%S")

    # Calculate the difference in seconds
    diff = abs((date_2 - date_1).total_seconds())

    # Increment the count for the corresponding difference
    if diff <= 60:
        counts[int(diff)] += 1

# Plot the results
x = list(counts.keys())
y = list(counts.values())

plt.figure(figsize=(10, 6))
plt.bar(x, y, color='skyblue', edgecolor = 'black', alpha=0.7)

plt.xlabel("Time Difference (seconds)")
plt.ylabel("Number of Images")
plt.title("Number of Images Taken Within 60 Seconds of Each Other")

plt.tight_layout()

# Save the plot
output_path = "/fs/ess/PAS2099/Vidhi/camera_trap_data/datasets/na/seq_plot.png"
plt.savefig(output_path, dpi=300)
plt.close()