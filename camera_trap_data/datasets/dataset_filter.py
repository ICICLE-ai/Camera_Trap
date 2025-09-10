class DatasetFilter:
    def __init__(self, metadata):
        """
        Initialize the DatasetFilter class with metadata.

        Args:
            metadata (dict): The metadata to filter.
        """
        self.metadata = metadata
        self.filtering_config = metadata.get("filtering_config", {})

    def apply_filters(self, global_config):
        """
        Apply all filters to the metadata based on the global and dataset-specific filtering configuration.
        Output a summary of the filtering process.

        Args:
            global_config (dict): Global filtering configuration.

        Returns:
            dict: Filtered metadata with dataset names and their filtered camera data.
        """
        filtering_summary = []

        # Iterate over each dataset in the metadata
        dataset_name = self.metadata["dataset_name"]
        dataset_data = self.metadata["data"]
        filtered_data = dataset_data  # Start with the original dataset data

        # Apply filters and track changes
        for filter_name, filter_enabled in global_config.items():
            if filter_enabled:
                filter_func = getattr(self, f"filter_{filter_name}", None)
                if filter_func:
                    before_count = len(filtered_data)
                    
                    from collections import defaultdict
                    before_by_location = defaultdict(int)
                    for entry in filtered_data:
                        location = entry["location"]
                        before_by_location[location] += 1

                    filtered_data = filter_func(filtered_data)
                    after_count = len(filtered_data)

                    after_by_location = defaultdict(int)
                    for entry in filtered_data:
                        location = entry["location"]
                        after_by_location[location] += 1

                    dropped_by_location = {
                        location: before_by_location[location] - after_by_location.get(location, 0)
                        for location in before_by_location
                    }

                    filtering_summary.append(
                        {
                            "dataset_name": dataset_name,
                            "filter_method": filter_name,
                            "original_count": before_count,
                            "filtered_count": after_count,
                            "dropped_by_location": dropped_by_location,
                        }
                    )

        # Print filtering summary
        self._print_filtering_summary(filtering_summary)

        return filtered_data

    def _print_filtering_summary(self, summary):
        """
        Print a summary of the filtering process.

        Args:
            summary (list): List of dictionaries containing filtering details.
        """
        print(f"{'Dataset Name':<15}{'Filter Method':<25}{'Original Count':<20}{'Filtered Count':<20}")
        print("-" * 80)

        current_dataset = None
        for entry in summary:
            dataset_name = entry['dataset_name']
            if dataset_name != current_dataset:
                print(f"{dataset_name:<15}", end="")
                current_dataset = dataset_name
            else:
                print(" " * 15, end="")  # Indent for subsequent rows of the same dataset
            print(
                f"{entry['filter_method']:<25}{entry['original_count']:<20}{entry['filtered_count']:<20}"
            )

            dropped_by_location = entry.get("dropped_by_location")
            if dropped_by_location:
                for location, count in dropped_by_location.items():
                    print(" " * 42 + f"- {location: <20}: {count} dropped")

        print("-" * 80)
        print()

    def filter_no_multi_classes(self, data):
        """Exclude entries with multiple classes in one image."""
        return [entry for entry in data if len(entry["class"]) <= 1]

    def filter_exclude_classes(self, data):
        """Exclude entries with specific class IDs."""
        classes_to_exclude = self.filtering_config.get("classes_to_exclude", [])
        return [
            entry
            for entry in data
            if all(cls["class_id"] not in classes_to_exclude for cls in entry["class"])
        ]

    def filter_no_bbox(self, data):
        """Exclude entries with no bounding boxes."""
        return [entry for entry in data if len(entry["bbox"]) > 0]

    def filter_no_low_bbox_conf(self, data):
        """Exclude entries with bounding boxes below a confidence threshold."""
        min_conf = self.filtering_config.get("bbox_min_conf", 0)
        filtered_data = []
        for entry in data:
            entry["bbox"] = [bbox for bbox in entry["bbox"] if any(float(b.get("conf", 0)) >= min_conf for b in [bbox])]
            if entry["bbox"]:  # Keep the entry only if it has valid bounding boxes
                filtered_data.append(entry)
        return filtered_data

    def filter_no_datetime(self, data):
        """Exclude entries with no datetime information."""
        return [entry for entry in data if entry.get("datetime") is not None and entry.get("datetime") != "" and str(entry.get("datetime")) != "nan"]

    def filter_only_animal_class_bbox(self, data):
        """Exclude entries where all bounding boxes are not of category 1 (non-animal)."""
        filtered_data = []
        for entry in data:
            entry["bbox"] = [bbox for bbox in entry["bbox"] if any(int(b.get("category", 0)) == 1 for b in [bbox])]
            if entry["bbox"]:  # Keep the entry only if it has valid bounding boxes
                filtered_data.append(entry)
        return filtered_data

    def filter_min_image_count(self, data):
        """Exclude cameras (locations) with fewer images than the minimum threshold."""
        min_count = self.filtering_config.get("min_image_count_threshold", 1000)  # Default threshold
        
        # Group images by location (camera)
        from collections import defaultdict
        location_counts = defaultdict(int)
        for entry in data:
            location_counts[entry["location"]] += 1
        
        # Filter out locations with fewer images than threshold
        valid_locations = {location for location, count in location_counts.items() if count >= min_count}
        
        # Return only entries from valid locations
        return [entry for entry in data if entry["location"] in valid_locations]

    def filter_no_missing_images(self, data):
        """Exclude entries where the source image file doesn't exist."""
        import os
        filtered_data = []
        missing_count = 0
        
        for entry in data:
            image_path = entry.get("image_path", "")
            if image_path and os.path.exists(image_path):
                filtered_data.append(entry)
            else:
                missing_count += 1
                if missing_count <= 10:  # Only print first 10 to avoid spam
                    print(f"Warning: Missing image file: {image_path}")
                elif missing_count == 11:
                    print(f"... (more missing images, suppressing further warnings)")
        
        if missing_count > 0:
            print(f"Total images filtered out due to missing files: {missing_count}")
        
        return filtered_data