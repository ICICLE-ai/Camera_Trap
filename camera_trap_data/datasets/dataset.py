import importlib
import yaml
import os
import json
import gzip
import hashlib
from glob import glob
from datasets.dataset_filter import DatasetFilter

class Dataset:
    def __init__(self, config_path, config=None):
        """
        Base class for managing datasets and cameras.

        Args:
            config_path (str): Path to the top-level configuration file.
            config (dict): Optional pre-loaded configuration dictionary. If None, loads from config_path.
        """
        self.config_path = config_path
        self.config = config
        self.metadata = {}  # To store the aggregated metadata
        self.global_filtering_config = {}  # To store global filtering configuration
        print("\n======================================== Loading Datasets =========================================\n")
        self._load_metadata()
        print("\n======================================== Filtering Datasets =======================================\n")
        self._apply_filters()
        self._reorganize_metadata()

    def _load_metadata(self):
        """
        Load all valid datasets and cameras based on the top-level configuration.
        """
        # Use pre-loaded config if available, otherwise parse from file
        if self.config is not None:
            config = self.config
        else:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        
        # Load global filtering configuration
        self.global_filtering_config = {
            "no_multi_classes": config.get("no_multi_classes", False),
            "exclude_classes": config.get("exclude_classes", False),
            "no_bbox": config.get("no_bbox", False),
            "no_low_bbox_conf": config.get("no_low_bbox_conf", False),
            "only_animal_class_bbox": config.get("only_animal_class_bbox", False),
            "no_datetime": config.get("no_datetime", False),
            "min_image_count": config.get("min_image_count", False),
            "no_missing_images": config.get("no_missing_images", False),
        }

        all_datasets = config.get("all_datasets", False)
        datasets_to_load = config.get("datasets", [])

        if all_datasets:            # Retrieve all dataset names from the configuration            
            # Exclude global keys and non-dataset sections like analysis settings
            excluded_keys = [
                "all_datasets", "datasets",
                "no_multi_classes", "exclude_classes", "no_bbox", "no_low_bbox_conf",
                "only_animal_class_bbox", "no_datetime", "min_image_count", "no_missing_images",
                "analysis_path", "metrics_analysis", "plot_analysis",
            ]
            datasets_to_load = [key for key in config.keys() if key not in excluded_keys]
            print("Loading all available datasets.")
        else:
            print(f"Loading datasets: {datasets_to_load}\n")

        # Iterate through the specified datasets
        for dataset_name in datasets_to_load:
            dataset_config = config.get(dataset_name, {})
            self._load_specific_dataset(dataset_name, dataset_config)

    def _load_specific_dataset(self, dataset_name, dataset_config):
        """
        Dynamically load a specific dataset and retrieve its metadata.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'safari').
            dataset_config (dict): Configuration for the dataset.
        """
        try:
            # Dynamically import the dataset-specific module
            module = importlib.import_module(f"datasets.{dataset_name}.{dataset_name}_dataset")
            # Convert snake_case to PascalCase for class name
            class_name = ''.join(word.capitalize() for word in dataset_name.split('_')) + "Dataset"
            dataset_class = getattr(module, class_name)

            # Determine dataset directory for caching
            dataset_dir = os.path.dirname(module.__file__)
            cache_payload = self._load_from_cache(dataset_name, dataset_dir, dataset_config)

            if cache_payload is None:
                # No cache or cache invalid; build fresh metadata via dataset module
                dataset_instance = dataset_class(dataset_config)
                dataset_metadata = dataset_instance.load_metadata()
                # Save to cache
                self._save_to_cache(dataset_name, dataset_dir, dataset_config, dataset_metadata)
            else:
                dataset_metadata = cache_payload

            # Append metadata to the global structure
            for key, value in dataset_metadata.items():
                self.metadata[key] = value

        except FileNotFoundError as e:
            # Skip datasets with missing metadata/bbox files, continue others
            print(f"⚠️  Skipping dataset '{dataset_name}': {e}")
        except ModuleNotFoundError:
            raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    def _get_metadata_signature(self, dataset_dir: str, dataset_config: dict) -> str:
        """Build a content signature based on metadata files and config to key the cache."""
        meta_dir = os.path.join(dataset_dir, "metadata")
        files = []
        if os.path.isdir(meta_dir):
            for path in sorted(glob(os.path.join(meta_dir, "*.json"))):
                try:
                    st = os.stat(path)
                    files.append({
                        "name": os.path.basename(path),
                        "size": st.st_size,
                        "mtime": int(st.st_mtime),
                    })
                except FileNotFoundError:
                    continue
        sig_obj = {
            "files": files,
            # Include dataset-specific filtering_config in case dataset modules derive fields from it
            "filtering_config": dataset_config.get("filtering_config", {}),
            # Version key to bust caches if format changes
            "cache_version": 1,
        }
        sig_str = json.dumps(sig_obj, sort_keys=True)
        return hashlib.sha1(sig_str.encode("utf-8")).hexdigest()

    def _cache_paths(self, dataset_dir: str, signature: str):
        cache_dir = os.path.join(dataset_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"metadata_{signature}.json.gz")
        latest = os.path.join(cache_dir, "metadata_latest.json.gz")
        return path, latest

    def _load_from_cache(self, dataset_name: str, dataset_dir: str, dataset_config: dict):
        """Try loading cached metadata for this dataset; return dict or None."""
        try:
            signature = self._get_metadata_signature(dataset_dir, dataset_config)
            cache_path, latest_path = self._cache_paths(dataset_dir, signature)
            path = cache_path if os.path.exists(cache_path) else (latest_path if os.path.exists(latest_path) else None)
            if not path:
                return None
            with gzip.open(path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            # Basic sanity check: expected key present
            if isinstance(payload, dict) and dataset_name in payload:
                print(f"🗂️  Loaded cache for '{dataset_name}' -> {os.path.relpath(path, dataset_dir)}\n")
                return payload
            return None
        except Exception:
            return None

    def _save_to_cache(self, dataset_name: str, dataset_dir: str, dataset_config: dict, dataset_metadata: dict):
        """Save dataset_metadata to gzip JSON cache keyed by signature."""
        try:
            signature = self._get_metadata_signature(dataset_dir, dataset_config)
            cache_path, latest_path = self._cache_paths(dataset_dir, signature)
            tmp_path = cache_path + ".tmp"
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(dataset_metadata, f)
            os.replace(tmp_path, cache_path)
            # Update latest pointer
            try:
                if os.path.exists(latest_path):
                    os.remove(latest_path)
            except Exception:
                pass
            try:
                os.link(cache_path, latest_path)
            except Exception:
                # Fallback to copy
                with gzip.open(cache_path, "rb") as src, gzip.open(latest_path, "wb") as dst:
                    dst.write(src.read())
            print(f"💾 Cached metadata for '{dataset_name}' -> {os.path.relpath(cache_path, dataset_dir)}\n")
        except Exception:
            # Do not fail the run on cache write issues
            pass

    def _apply_filters(self):
        """
        Apply filters to the metadata based on the global and dataset-specific filtering configuration.
        """
        for dataset_name, dataset_info in self.metadata.items():
            # Initialize DatasetFilter with metadata
            dataset_filter = DatasetFilter(metadata=dataset_info)

            # Apply filters based on the global and dataset-specific filtering configuration
            dataset_info["data"] = dataset_filter.apply_filters(global_config=self.global_filtering_config)

    def _reorganize_metadata(self):
        """
        Reorganize metadata to group data by camera under each dataset.
        The final structure will look like:
        {
            "dataset_name": {
                "camera_name": {
                    "data": [entries belonging to this camera]
                }
            }
        }
        """
        reorganized_metadata = {}

        for dataset_name, dataset_info in self.metadata.items():
            dataset_cameras = {}
            for entry in dataset_info["data"]:
                camera_name = entry["location"]  # Use 'location' to group by camera
                if camera_name not in dataset_cameras:
                    dataset_cameras[camera_name] = {"data": []}
                dataset_cameras[camera_name]["data"].append(entry)

            reorganized_metadata[dataset_name] = dataset_cameras

        self.metadata = reorganized_metadata

    def get_data_length(self):
        """Return the total number of data entries across all datasets."""
        return sum(len(dataset["data"]) for dataset in self.metadata.values())

    def get_animal_classes(self):
        """Return a list of unique animal classes across all datasets."""
        classes = set()
        for dataset in self.metadata.values():
            classes.update(entry["class"] for entry in dataset["data"])
        return list(classes)

    def get_image_ids(self):
        """Return a list of all image IDs across all datasets."""
        image_ids = []
        for dataset in self.metadata.values():
            image_ids.extend(entry["image_id"] for entry in dataset["data"])
        return image_ids
    
    def get_node_data(self, node_num, total_nodes=100):
        """
        Get data for a specific node number by chunking the data.
        
        This method divides all data entries across cameras and datasets into chunks
        and returns only the chunk assigned to the specified node.

        Args:
            node_num (int): The node number to filter data by (0-based indexing).
            total_nodes (int): Total number of nodes for distributed processing (default: 100).

        Returns:
            Dataset: A new Dataset instance containing only the data chunk for this node.
        """
        import copy
        
        # Collect all data entries with their source information
        all_entries = []
        for dataset_name, dataset_cameras in self.metadata.items():
            for camera_name, camera_data in dataset_cameras.items():
                for entry in camera_data["data"]:
                    all_entries.append({
                        "dataset": dataset_name,
                        "camera": camera_name,
                        "entry": entry
                    })
        
        total_entries = len(all_entries)
        
        # Calculate chunk size and boundaries
        chunk_size = total_entries // total_nodes
        remainder = total_entries % total_nodes
        
        # Calculate start and end indices for this node
        if node_num < remainder:
            # First 'remainder' nodes get one extra entry
            start_idx = node_num * (chunk_size + 1)
            end_idx = start_idx + chunk_size + 1
        else:
            # Remaining nodes get the standard chunk size
            start_idx = remainder * (chunk_size + 1) + (node_num - remainder) * chunk_size
            end_idx = start_idx + chunk_size
        
        # Get the chunk of data for this node
        node_entries = all_entries[start_idx:end_idx]
        
        print(f"Node {node_num}: Processing {len(node_entries)} entries out of {total_entries} total entries")
        print(f"Node {node_num}: Data range [{start_idx}:{end_idx}]")
        
        # Create a new Dataset instance with filtered metadata
        filtered_dataset = copy.deepcopy(self)
        filtered_metadata = {}
        
        # Reorganize the node entries back into the metadata structure
        for entry_info in node_entries:
            dataset_name = entry_info["dataset"]
            camera_name = entry_info["camera"]
            entry = entry_info["entry"]
            
            # Initialize dataset if not exists
            if dataset_name not in filtered_metadata:
                filtered_metadata[dataset_name] = {}
            
            # Initialize camera if not exists
            if camera_name not in filtered_metadata[dataset_name]:
                filtered_metadata[dataset_name][camera_name] = {"data": []}
            
            # Add the entry to the filtered metadata
            filtered_metadata[dataset_name][camera_name]["data"].append(entry)
        
        # Update the filtered dataset's metadata
        filtered_dataset.metadata = filtered_metadata
        
        return filtered_dataset