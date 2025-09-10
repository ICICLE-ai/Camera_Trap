from collections import defaultdict
import os
import csv
from datetime import datetime
import math

# Define how each metric should be saved
METRIC_SAVE_CONFIG = {
    # Existing metrics only
    "ts_l1_accumulated": "average",
    "ts_l1_full": "average",
    "gini_index": "value",
    "l1_test": "value",
}

class MetricAnalysis:
    def __init__(self, config, dataset, analysis_path, datetime_formats=None, checkpoint_days=30):
        """
        Initialize the MetricAnalysis module.

        Args:
            config (dict): Configuration dictionary.
            dataset (Dataset): Prepared dataset object.
            analysis_path (str): Root path for saving analysis results.
            datetime_formats (list): List of datetime format strings to try when parsing dates.
            checkpoint_days (int): Number of days for each checkpoint interval.
        """
        self.config = config.get("metrics_analysis", {})
        self.categorization_thresholds = {}
        self.difficulty_rating_config = {}
        self.dataset = dataset
        self.analysis_path = analysis_path
        self.datetime_formats = datetime_formats or ["%Y:%m:%d %H:%M:%S"]  # Default fallback
        self.checkpoint_days = checkpoint_days

    def run(self):
        """
        Run all metric-related analyses based on the configuration.
        """
        # Existing metrics
        if self.config.get("ts_l1_accumulated", False):
            self._compute_ts_l1_accumulated()
        if self.config.get("ts_l1_full", False):
            self._compute_ts_l1_full()
        if self.config.get("gini_index", False):
            self._compute_gini_index()
        if self.config.get("l1_test", False):
            self._compute_l1_test()
        
    # Save results
        self._save_metrics_results()

    def _compute_ts_l1_accumulated(self):
        """
        Compute the accumulated L1 metric (time-series perspective) and store the results in the metadata.
        Train data is accumulated up to the previous checkpoint, and test data is from the current checkpoint.
        """
        print("▶️ Computing time-series L1 accumulated metric...")

        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue  # Skip if no checkpoints are available

                cumulative_train_data = []
                analysis_results = {}
                total_l1_metric = 0
                total_class_count = 0

                sorted_ckp_ids = sorted(camera_data["ckp"].keys())

                for idx, ckp_id in enumerate(sorted_ckp_ids):
                    ckp_data = camera_data["ckp"][ckp_id]

                    if idx == 0:
                        continue  # Skip the first checkpoint

                    prev_ckp_id = sorted_ckp_ids[idx - 1]
                    cumulative_train_data.extend(camera_data["ckp"][prev_ckp_id]["train"])
                    train_data = cumulative_train_data
                    test_data = ckp_data["val"]

                    train_counts = self._calculate_class_counts(train_data)
                    test_counts = self._calculate_class_counts(test_data)

                    train_total = sum(train_counts.values())
                    test_total = sum(test_counts.values())

                    l1_metric = 0

                    for cls, test_count in test_counts.items():
                        q = test_count / test_total if test_total > 0 else 0
                        train_count = train_counts.get(cls, 0)
                        p = train_count / train_total if train_total > 0 else 0

                        if q > p:
                            l1_metric += abs(q - p)
                            total_l1_metric += abs(q - p)
                            total_class_count += 1

                    analysis_results[ckp_id] = {"ts_l1_accumulated": l1_metric}

                global_avg_l1_metric = total_l1_metric / total_class_count if total_class_count > 0 else 0

                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["ts_l1_accumulated"] = {
                    "checkpoints": analysis_results,
                    "average": global_avg_l1_metric,
                }

    def _compute_ts_l1_full(self):
        """
        Compute the accumulated L1 metric using all train checkpoints for every test checkpoint.
        """
        print("▶️ Computing time-series L1 full metric...")

        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue  # Skip if no checkpoints are available

                full_train_data = []
                for ckp_data in camera_data["ckp"].values():
                    full_train_data.extend(ckp_data["train"])

                analysis_results = {}
                total_l1_metric = 0
                total_class_count = 0

                sorted_ckp_ids = sorted(camera_data["ckp"].keys())

                for ckp_id in sorted_ckp_ids:
                    ckp_data = camera_data["ckp"][ckp_id]
                    train_data = full_train_data
                    test_data = ckp_data["val"]

                    train_counts = self._calculate_class_counts(train_data)
                    test_counts = self._calculate_class_counts(test_data)

                    train_total = sum(train_counts.values())
                    test_total = sum(test_counts.values())

                    l1_metric = 0

                    for cls, test_count in test_counts.items():
                        q = test_count / test_total if test_total > 0 else 0
                        train_count = train_counts.get(cls, 0)
                        p = train_count / train_total if train_total > 0 else 0

                        if q > p:
                            l1_metric += abs(q - p)
                            total_l1_metric += abs(q - p)
                            total_class_count += 1

                    analysis_results[ckp_id] = {"ts_l1_full": l1_metric}

                global_avg_l1_metric = total_l1_metric / total_class_count if total_class_count > 0 else 0

                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["ts_l1_full"] = {
                    "checkpoints": analysis_results,
                    "average": global_avg_l1_metric,
                }

    def _compute_gini_index(self):
        """
        Compute the Gini index for the class distribution across all checkpoints (combined train and test data).
        """
        print("▶️ Computing Gini index...")

        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue  # Skip if no checkpoints are available

                # Aggregate class counts across all checkpoints (train + test)
                combined_class_counts = defaultdict(int)
                for ckp_data in camera_data["ckp"].values():
                    for entry in ckp_data["train"] + ckp_data["val"]:
                        for cls in entry["class"]:
                            combined_class_counts[cls["class_id"]] += 1

                # Calculate total samples
                total_samples = sum(combined_class_counts.values())

                # Calculate Gini index
                if total_samples == 0:
                    gini_index = 0
                else:
                    gini_index = 1 - sum((count / total_samples) ** 2 for count in combined_class_counts.values())

                # Store the Gini index in the camera's metadata
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["gini_index"] = {"value": gini_index}

    def _compute_l1_test(self):
        """
        Compute the L1 drift metric across adjacent checkpoints and store the result in the metadata.
        This measures how much class distributions change between adjacent checkpoints.
        Normalization is performed within each checkpoint (local normalization).
        """
        print("▶️ Computing L1 drift metric with local normalization...")

        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue  # Skip if no checkpoints are available

                sorted_ckp_ids = sorted(camera_data["ckp"].keys())
                if len(sorted_ckp_ids) < 2:
                    # L1 drift requires at least two checkpoints
                    continue

                total_l1_drift = 0
                num_pairs = 0

                for i in range(len(sorted_ckp_ids) - 1):
                    ckp_id_1 = sorted_ckp_ids[i]
                    ckp_id_2 = sorted_ckp_ids[i + 1]

                    # Get class counts for the two checkpoints
                    ckp_1_data = camera_data["ckp"][ckp_id_1]["val"]
                    ckp_2_data = camera_data["ckp"][ckp_id_2]["val"]

                    counts_1 = self._calculate_class_counts(ckp_1_data)
                    counts_2 = self._calculate_class_counts(ckp_2_data)

                    # Normalize class frequencies within each checkpoint
                    total_1 = sum(counts_1.values())
                    total_2 = sum(counts_2.values())

                    normalized_1 = {cls: count / total_1 for cls, count in counts_1.items() if total_1 > 0}
                    normalized_2 = {cls: count / total_2 for cls, count in counts_2.items() if total_2 > 0}

                    # Get the union of all classes
                    all_classes = set(normalized_1.keys()).union(set(normalized_2.keys()))

                    # Compute L1 drift for this pair of checkpoints
                    l1_drift = sum(abs(normalized_1.get(cls, 0) - normalized_2.get(cls, 0)) for cls in all_classes)
                    total_l1_drift += l1_drift
                    num_pairs += 1

                # Calculate the average L1 drift across all pairs of adjacent checkpoints
                average_l1_drift = total_l1_drift / num_pairs if num_pairs > 0 else 0

                # Store the result in the camera's metadata
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["l1_test"] = {"value": average_l1_drift}

    def _compute_temporal_variance(self):
        """
        Compute temporal variance - standard deviation of class distributions across checkpoints.
        """
        print("▶️ Computing temporal variance...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Get class proportions for each checkpoint
                checkpoint_proportions = []
                for ckp_data in camera_data["ckp"].values():
                    val_data = ckp_data["val"]
                    class_counts = self._calculate_class_counts(val_data)
                    total_count = sum(class_counts.values())
                    
                    if total_count > 0:
                        proportions = {cls: count / total_count for cls, count in class_counts.items()}
                        checkpoint_proportions.append(proportions)
                
                # Calculate variance across checkpoints
                if len(checkpoint_proportions) < 2:
                    temporal_variance = 0
                else:
                    # Get all unique classes
                    all_classes = set()
                    for props in checkpoint_proportions:
                        all_classes.update(props.keys())
                    
                    # Calculate variance for each class, then average
                    class_variances = []
                    for cls in all_classes:
                        cls_proportions = [props.get(cls, 0) for props in checkpoint_proportions]
                        if len(cls_proportions) > 1:
                            mean_prop = sum(cls_proportions) / len(cls_proportions)
                            variance = sum((p - mean_prop) ** 2 for p in cls_proportions) / len(cls_proportions)
                            class_variances.append(variance)
                    
                    temporal_variance = sum(class_variances) / len(class_variances) if class_variances else 0
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["temporal_variance"] = {"value": temporal_variance}
    
    def _compute_checkpoint_consistency(self):
        """
        Compute checkpoint consistency - how similar adjacent checkpoints are.
        """
        print("▶️ Computing checkpoint consistency...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                sorted_ckp_ids = sorted(camera_data["ckp"].keys())
                if len(sorted_ckp_ids) < 2:
                    consistency = 1.0
                else:
                    similarities = []
                    for i in range(len(sorted_ckp_ids) - 1):
                        ckp1_data = camera_data["ckp"][sorted_ckp_ids[i]]["val"]
                        ckp2_data = camera_data["ckp"][sorted_ckp_ids[i + 1]]["val"]
                        
                        counts1 = self._calculate_class_counts(ckp1_data)
                        counts2 = self._calculate_class_counts(ckp2_data)
                        
                        total1 = sum(counts1.values())
                        total2 = sum(counts2.values())
                        
                        if total1 > 0 and total2 > 0:
                            props1 = {cls: count / total1 for cls, count in counts1.items()}
                            props2 = {cls: count / total2 for cls, count in counts2.items()}
                            
                            # Calculate cosine similarity
                            all_classes = set(props1.keys()).union(set(props2.keys()))
                            dot_product = sum(props1.get(cls, 0) * props2.get(cls, 0) for cls in all_classes)
                            norm1 = math.sqrt(sum(props1.get(cls, 0) ** 2 for cls in all_classes))
                            norm2 = math.sqrt(sum(props2.get(cls, 0) ** 2 for cls in all_classes))
                            
                            if norm1 > 0 and norm2 > 0:
                                similarity = dot_product / (norm1 * norm2)
                                similarities.append(similarity)
                    
                    consistency = sum(similarities) / len(similarities) if similarities else 1.0
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["checkpoint_consistency"] = {"value": consistency}
    
    def _compute_seasonal_drift(self):
        """
        Compute seasonal drift - variance in class distributions across different time periods.
        """
        print("▶️ Computing seasonal drift...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Group checkpoints by season (quarter)
                seasonal_data = defaultdict(list)
                
                for ckp_data in camera_data["ckp"].values():
                    for entry in ckp_data["val"]:
                        dt_str = entry.get("datetime")
                        if dt_str:
                            try:
                                dt = self._parse_datetime(dt_str)
                                season = (dt.month - 1) // 3  # 0-3 for quarters
                                seasonal_data[season].append(entry)
                            except:
                                pass
                
                # Calculate class distributions for each season
                seasonal_proportions = []
                for season_entries in seasonal_data.values():
                    if season_entries:
                        class_counts = self._calculate_class_counts(season_entries)
                        total_count = sum(class_counts.values())
                        if total_count > 0:
                            proportions = {cls: count / total_count for cls, count in class_counts.items()}
                            seasonal_proportions.append(proportions)
                
                # Calculate variance across seasons
                if len(seasonal_proportions) < 2:
                    seasonal_drift = 0
                else:
                    all_classes = set()
                    for props in seasonal_proportions:
                        all_classes.update(props.keys())
                    
                    class_variances = []
                    for cls in all_classes:
                        cls_proportions = [props.get(cls, 0) for props in seasonal_proportions]
                        if len(cls_proportions) > 1:
                            mean_prop = sum(cls_proportions) / len(cls_proportions)
                            variance = sum((p - mean_prop) ** 2 for p in cls_proportions) / len(cls_proportions)
                            class_variances.append(variance)
                    
                    seasonal_drift = sum(class_variances) / len(class_variances) if class_variances else 0
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["seasonal_drift"] = {"value": seasonal_drift}
    
    def _compute_shannon_entropy(self):
        """
        Compute Shannon entropy - measures diversity of class distributions.
        """
        print("▶️ Computing Shannon entropy...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Aggregate class counts across all checkpoints
                combined_class_counts = defaultdict(int)
                for ckp_data in camera_data["ckp"].values():
                    for entry in ckp_data["train"] + ckp_data["val"]:
                        for cls in entry["class"]:
                            combined_class_counts[cls["class_id"]] += 1
                
                total_samples = sum(combined_class_counts.values())
                
                if total_samples == 0:
                    shannon_entropy = 0
                else:
                    shannon_entropy = -sum(
                        (count / total_samples) * math.log2(count / total_samples)
                        for count in combined_class_counts.values()
                        if count > 0
                    )
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["shannon_entropy"] = {"value": shannon_entropy}
    
    def _compute_effective_num_classes(self):
        """
        Compute effective number of classes - exp(Shannon_entropy).
        """
        print("▶️ Computing effective number of classes...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "analysis" in camera_data and "shannon_entropy" in camera_data["analysis"]:
                    shannon_entropy = camera_data["analysis"]["shannon_entropy"]["value"]
                    effective_num_classes = math.exp(shannon_entropy) if shannon_entropy > 0 else 1
                    
                    if "analysis" not in camera_data:
                        camera_data["analysis"] = {}
                    camera_data["analysis"]["effective_num_classes"] = {"value": effective_num_classes}
    
    def _compute_class_balance_ratio(self):
        """
        Compute class balance ratio - ratio of most frequent to least frequent class.
        """
        print("▶️ Computing class balance ratio...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Aggregate class counts across all checkpoints
                combined_class_counts = defaultdict(int)
                for ckp_data in camera_data["ckp"].values():
                    for entry in ckp_data["train"] + ckp_data["val"]:
                        for cls in entry["class"]:
                            combined_class_counts[cls["class_id"]] += 1
                
                if not combined_class_counts:
                    class_balance_ratio = 1
                else:
                    max_count = max(combined_class_counts.values())
                    min_count = min(combined_class_counts.values())
                    class_balance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["class_balance_ratio"] = {"value": class_balance_ratio}
    
    def _compute_tail_heaviness(self):
        """
        Compute tail heaviness - percentage of images in classes representing <5% of total data.
        """
        print("▶️ Computing tail heaviness...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Aggregate class counts across all checkpoints
                combined_class_counts = defaultdict(int)
                for ckp_data in camera_data["ckp"].values():
                    for entry in ckp_data["train"] + ckp_data["val"]:
                        for cls in entry["class"]:
                            combined_class_counts[cls["class_id"]] += 1
                
                total_samples = sum(combined_class_counts.values())
                
                if total_samples == 0:
                    tail_heaviness = 0
                else:
                    tail_samples = sum(
                        count for count in combined_class_counts.values()
                        if count / total_samples < 0.05
                    )
                    tail_heaviness = tail_samples / total_samples
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["tail_heaviness"] = {"value": tail_heaviness}
    
    def _compute_checkpoint_diversity(self):
        """
        Compute checkpoint diversity - number of unique class combinations across checkpoints.
        """
        print("▶️ Computing checkpoint diversity...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                unique_combinations = set()
                for ckp_data in camera_data["ckp"].values():
                    val_data = ckp_data["val"]
                    classes_in_ckp = set()
                    for entry in val_data:
                        for cls in entry["class"]:
                            classes_in_ckp.add(cls["class_id"])
                    
                    # Convert to frozenset for hashing
                    unique_combinations.add(frozenset(classes_in_ckp))
                
                num_checkpoints = len(camera_data["ckp"])
                checkpoint_diversity = len(unique_combinations) / num_checkpoints if num_checkpoints > 0 else 0
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["checkpoint_diversity"] = {"value": checkpoint_diversity}
    
    def _compute_temporal_sparsity(self):
        """
        Compute temporal sparsity - percentage of checkpoints where classes appear/disappear.
        """
        print("▶️ Computing temporal sparsity...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Track class presence across checkpoints
                class_presence = defaultdict(list)
                sorted_ckp_ids = sorted(camera_data["ckp"].keys())
                
                for ckp_id in sorted_ckp_ids:
                    ckp_data = camera_data["ckp"][ckp_id]
                    classes_in_ckp = set()
                    for entry in ckp_data["val"]:
                        for cls in entry["class"]:
                            classes_in_ckp.add(cls["class_id"])
                    
                    # Record presence/absence for each class
                    all_classes = set()
                    for ckp_data in camera_data["ckp"].values():
                        for entry in ckp_data["train"] + ckp_data["val"]:
                            for cls in entry["class"]:
                                all_classes.add(cls["class_id"])
                    
                    for cls in all_classes:
                        class_presence[cls].append(1 if cls in classes_in_ckp else 0)
                
                # Calculate sparsity
                total_transitions = 0
                total_possible_transitions = 0
                
                for cls, presence_list in class_presence.items():
                    if len(presence_list) > 1:
                        transitions = sum(1 for i in range(len(presence_list) - 1) 
                                        if presence_list[i] != presence_list[i + 1])
                        total_transitions += transitions
                        total_possible_transitions += len(presence_list) - 1
                
                temporal_sparsity = total_transitions / total_possible_transitions if total_possible_transitions > 0 else 0
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["temporal_sparsity"] = {"value": temporal_sparsity}
    
    def _compute_class_persistence(self):
        """
        Compute class persistence - how consistently classes appear across checkpoints.
        """
        print("▶️ Computing class persistence...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Get all unique classes
                all_classes = set()
                for ckp_data in camera_data["ckp"].values():
                    for entry in ckp_data["train"] + ckp_data["val"]:
                        for cls in entry["class"]:
                            all_classes.add(cls["class_id"])
                
                if not all_classes:
                    class_persistence = 1.0
                else:
                    # Calculate persistence for each class
                    class_persistences = []
                    num_checkpoints = len(camera_data["ckp"])
                    
                    for cls in all_classes:
                        appearances = 0
                        for ckp_data in camera_data["ckp"].values():
                            class_in_ckp = any(cls["class_id"] == cls for entry in ckp_data["val"] for cls in entry["class"])
                            if class_in_ckp:
                                appearances += 1
                        
                        persistence = appearances / num_checkpoints
                        class_persistences.append(persistence)
                    
                    class_persistence = sum(class_persistences) / len(class_persistences)
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["class_persistence"] = {"value": class_persistence}
    
    def _compute_temporal_coverage(self):
        """
        Compute temporal coverage - time span vs number of checkpoints (density).
        """
        print("▶️ Computing temporal coverage...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Get time span
                timestamps = []
                for ckp in camera_data["ckp"].values():
                    for entry in ckp["train"] + ckp["val"]:
                        dt_str = entry.get("datetime")
                        if dt_str:
                            try:
                                timestamps.append(self._parse_datetime(dt_str))
                            except:
                                pass
                
                if len(timestamps) < 2:
                    temporal_coverage = 0
                else:
                    min_time = min(timestamps)
                    max_time = max(timestamps)
                    time_span_days = (max_time - min_time).days
                    num_checkpoints = len(camera_data["ckp"])
                    
                    # Coverage as checkpoints per day
                    temporal_coverage = num_checkpoints / time_span_days if time_span_days > 0 else 0
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["temporal_coverage"] = {"value": temporal_coverage}
    
    def _compute_class_emergence_rate(self):
        """
        Compute class emergence rate - how often new classes appear in later checkpoints.
        """
        print("▶️ Computing class emergence rate...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                sorted_ckp_ids = sorted(camera_data["ckp"].keys())
                if len(sorted_ckp_ids) < 2:
                    class_emergence_rate = 0
                else:
                    seen_classes = set()
                    new_classes_count = 0
                    
                    for ckp_id in sorted_ckp_ids:
                        ckp_classes = set()
                        for entry in camera_data["ckp"][ckp_id]["val"]:
                            for cls in entry["class"]:
                                ckp_classes.add(cls["class_id"])
                        
                        new_classes = ckp_classes - seen_classes
                        new_classes_count += len(new_classes)
                        seen_classes.update(ckp_classes)
                    
                    total_classes = len(seen_classes)
                    class_emergence_rate = new_classes_count / total_classes if total_classes > 0 else 0
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["class_emergence_rate"] = {"value": class_emergence_rate}
    
    def _compute_data_density(self):
        """
        Compute data density - images per month/checkpoint.
        """
        print("▶️ Computing data density...")
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Count total unique images
                unique_images = set()
                for ckp in camera_data["ckp"].values():
                    for entry in ckp["train"] + ckp["val"]:
                        unique_images.add(entry["image_id"])
                
                total_images = len(unique_images)
                
                # Get time span in months
                timestamps = []
                for ckp in camera_data["ckp"].values():
                    for entry in ckp["train"] + ckp["val"]:
                        dt_str = entry.get("datetime")
                        if dt_str:
                            try:
                                timestamps.append(self._parse_datetime(dt_str))
                            except:
                                pass
                
                if timestamps:
                    min_time = min(timestamps)
                    max_time = max(timestamps)
                    time_span_months = (max_time.year - min_time.year) * 12 + (max_time.month - min_time.month)
                    time_span_months = max(1, time_span_months)  # Avoid division by zero
                    
                    data_density = total_images / time_span_months
                else:
                    data_density = 0
                
                if "analysis" not in camera_data:
                    camera_data["analysis"] = {}
                camera_data["analysis"]["data_density"] = {"value": data_density}
    
    def _parse_datetime(self, dt_str):
        """
        Parse datetime string using available formats.
        """
        for fmt in self.datetime_formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except (ValueError, TypeError):
                continue
        raise ValueError(f"Unknown datetime format: {dt_str}")
    
    def _calculate_difficulty_rating(self, camera_data):
        """
        Calculate continuous difficulty rating (0-1) based on computed metrics.
        
        Args:
            camera_data (dict): Camera data containing analysis results
            
        Returns:
            tuple: (difficulty_score, difficulty_level, component_scores)
        """
        if "analysis" not in camera_data:
            return 0.5, "moderate", {}
        
        analysis = camera_data["analysis"]
        
        # Check if difficulty rating is enabled
        if not self.difficulty_rating_config.get("enabled", False):
            # Fall back to legacy categorization converted to score
            category = self._categorize_camera(camera_data)
            legacy_score = self._category_to_score(category)
            return legacy_score, self._score_to_level(legacy_score), {}
        
        # Get configuration
        metric_weights = self.difficulty_rating_config.get("metric_weights", {})
        normalization_ranges = self.difficulty_rating_config.get("normalization_ranges", {})
        
        # Calculate normalized scores for each metric
        component_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, weight in metric_weights.items():
            if metric_name in METRIC_SAVE_CONFIG:
                # Get metric value
                metric_value = analysis.get(metric_name, {}).get(METRIC_SAVE_CONFIG[metric_name], None)
                
                if metric_value is not None and not math.isnan(metric_value):
                    # Handle infinite values
                    if math.isinf(metric_value):
                        metric_value = 1000.0 if metric_value > 0 else 0.0
                    
                    # Normalize metric to 0-1 range
                    norm_range = normalization_ranges.get(metric_name, [0.0, 1.0])
                    min_val, max_val = norm_range
                    
                    # Clamp and normalize
                    normalized_value = max(0.0, min(1.0, (metric_value - min_val) / (max_val - min_val)))
                    
                    # Store component score
                    component_scores[metric_name] = {
                        'raw_value': metric_value,
                        'normalized_value': normalized_value,
                        'weight': weight,
                        'contribution': normalized_value * abs(weight)
                    }
                    
                    # Add to weighted sum
                    weighted_sum += normalized_value * weight
                    total_weight += abs(weight)
        
        # Calculate final difficulty score
        if total_weight > 0:
            difficulty_score = weighted_sum / total_weight
            # Ensure score is in [0, 1] range
            difficulty_score = max(0.0, min(1.0, difficulty_score))
        else:
            difficulty_score = 0.5  # Default moderate difficulty
        
        # Determine difficulty level
        difficulty_level = self._score_to_level(difficulty_score)
        
        return difficulty_score, difficulty_level, component_scores
    
    def _score_to_level(self, score):
        """Convert difficulty score to human-readable level."""
        difficulty_levels = self.difficulty_rating_config.get("difficulty_levels", {
            "very_easy": [0.0, 0.2],
            "easy": [0.2, 0.4],
            "moderate": [0.4, 0.6],
            "hard": [0.6, 0.8],
            "very_hard": [0.8, 1.0]
        })
        
        for level, (min_score, max_score) in difficulty_levels.items():
            if min_score <= score <= max_score:
                return level
        return "moderate"
    
    def _category_to_score(self, category):
        """Convert legacy category to approximate difficulty score."""
        if not category or category == "Unknown":
            return 0.5
        
        categories = [cat.strip() for cat in category.split(';')]
        
        # Base score
        score = 0.3  # Start with moderate
        
        # Add difficulty for each category
        if "Hard" in categories:
            score += 0.25
        if "Long-tailed" in categories:
            score += 0.2
        if "Temporally Complex" in categories:
            score += 0.15
        if "Easy" in categories:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _categorize_camera(self, camera_data):
        """
        Legacy categorization method (kept for backward compatibility).
        """
        if "analysis" not in camera_data:
            return "Unknown"
        
        analysis = camera_data["analysis"]
        categories = []
        
        # Check for Easy vs Hard
        l1_test = analysis.get("l1_test", {}).get("value", 0)
        class_persistence = analysis.get("class_persistence", {}).get("value", 1)
        gini_index = analysis.get("gini_index", {}).get("value", 0)
        
        temporal_drift_threshold = self.categorization_thresholds.get("temporal_drift_threshold", 0.3)
        class_persistence_threshold = self.categorization_thresholds.get("class_persistence_threshold", 0.7)
        gini_threshold = self.categorization_thresholds.get("gini_threshold", 0.6)
        
        if (l1_test < temporal_drift_threshold and 
            class_persistence > class_persistence_threshold and 
            gini_index < gini_threshold):
            categories.append("Easy")
        elif (l1_test > temporal_drift_threshold or 
              class_persistence < class_persistence_threshold or 
              gini_index > gini_threshold):
            categories.append("Hard")
        
        # Check for Long-tailed
        tail_heaviness = analysis.get("tail_heaviness", {}).get("value", 0)
        class_balance_ratio = analysis.get("class_balance_ratio", {}).get("value", 1)
        
        tail_heaviness_threshold = self.categorization_thresholds.get("tail_heaviness_threshold", 0.8)
        class_balance_ratio_threshold = self.categorization_thresholds.get("class_balance_ratio_threshold", 10)
        
        if (tail_heaviness > tail_heaviness_threshold or 
            class_balance_ratio > class_balance_ratio_threshold):
            categories.append("Long-tailed")
        
        # Check for Temporally Complex
        checkpoint_diversity = analysis.get("checkpoint_diversity", {}).get("value", 0)
        temporal_sparsity = analysis.get("temporal_sparsity", {}).get("value", 0)
        seasonal_drift = analysis.get("seasonal_drift", {}).get("value", 0)
        
        checkpoint_diversity_threshold = self.categorization_thresholds.get("checkpoint_diversity_threshold", 0.7)
        temporal_sparsity_threshold = self.categorization_thresholds.get("temporal_sparsity_threshold", 0.5)
        seasonal_drift_threshold = self.categorization_thresholds.get("seasonal_drift_threshold", 0.4)
        
        if (checkpoint_diversity > checkpoint_diversity_threshold or 
            temporal_sparsity > temporal_sparsity_threshold or 
            seasonal_drift > seasonal_drift_threshold):
            categories.append("Temporally Complex")
        
        return "; ".join(categories) if categories else "Standard"
    
    def _calculate_class_counts(self, data):
        """
        Calculate class counts for a given set of data entries.

        Args:
            data (list): List of data entries for a camera.

        Returns:
            dict: Class counts.
        """
        counts = defaultdict(int)
        for entry in data:
            for cls in entry["class"]:
                counts[cls["class_id"]] += 1
        return counts

    def _save_metrics_results(self):
        """
        Save each camera's basic info and metrics to CSV files:
        1. Global CSV file with all datasets in analysis_path
        2. Dataset-specific CSV files in analysis_path/{dataset_name}/
        """
        print("💾 Saving metrics results to CSV...")
        
        # Prepare headers
        headers = [
            "Dataset",
            "Camera Name",
            "Total Images",
            "Total Unique Classes",
            "Time Span (Months)",
            "Number of Checkpoints",
        ]

        # Add metric headers for enabled metrics
        metric_headers = []
        for metric_name in self.config:
            if self.config[metric_name]:
                metric_headers.append(metric_name)

        headers.extend(metric_headers)

        # Collect all data first
        all_data = []
        dataset_data = defaultdict(list)
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue

                # Calculate basic info
                unique_images = set(entry["image_id"] for ckp in camera_data["ckp"].values() for entry in ckp["train"] + ckp["val"])
                total_images = len(unique_images)                    
                unique_classes = set(cls["class_id"] for ckp in camera_data["ckp"].values() for entry in ckp["train"] + ckp["val"] for cls in entry["class"])
                total_unique_classes = len(unique_classes)

                # Calculate time span
                timestamps = []
                for ckp in camera_data["ckp"].values():
                    for entry in ckp["train"] + ckp["val"]:
                        dt_str = entry.get("datetime")
                        if dt_str:
                            try:
                                timestamps.append(self._parse_datetime(dt_str))
                            except ValueError:
                                pass  # Skip unparseable datetimes
                
                if timestamps:
                    min_time = min(timestamps)
                    max_time = max(timestamps)
                    time_span_months = (max_time.year - min_time.year) * 12 + (max_time.month - min_time.month)
                else:
                    time_span_months = 0

                num_checkpoints = len(camera_data["ckp"])
                
                # Build row
                row = {
                    "Dataset": dataset_name,
                    "Camera Name": camera_name,
                    "Total Images": total_images,
                    "Total Unique Classes": total_unique_classes,
                    "Time Span (Months)": time_span_months,
                    "Number of Checkpoints": num_checkpoints,
                }

                # Add metric values
                for metric_name in metric_headers:
                    metric_value = camera_data.get("analysis", {}).get(metric_name, {}).get(METRIC_SAVE_CONFIG[metric_name], "N/A")
                    if isinstance(metric_value, float):
                        metric_value = round(metric_value, 4)
                    row[metric_name] = metric_value

                all_data.append(row)
                dataset_data[dataset_name].append(row)

        # Save global CSV file (all datasets)
        global_output_path = os.path.join(self.analysis_path, f"metrics_results_{self.checkpoint_days}days.csv")
        self._write_csv_file(global_output_path, headers, all_data)
        print(f"✅ Global metrics results saved to: {global_output_path}")

        # Save dataset-specific CSV files
        for dataset_name, dataset_rows in dataset_data.items():
            dataset_dir = os.path.join(self.analysis_path, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset_output_path = os.path.join(dataset_dir, f"metrics_results_{self.checkpoint_days}days.csv")
            self._write_csv_file(dataset_output_path, headers, dataset_rows)
            print(f"✅ Dataset '{dataset_name}' metrics results saved to: {dataset_output_path}")
        
    # Difficulty breakdown removed

    def _write_csv_file(self, output_path, headers, data):
        """
        Helper method to write CSV file with given headers and data.
        
        Args:
            output_path (str): Path to output CSV file
            headers (list): List of column headers
            data (list): List of dictionaries with data rows
        """
        with open(output_path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    
    def _save_difficulty_breakdown(self):
        """
        Save detailed difficulty breakdown showing component scores:
        1. Global CSV file with all datasets in analysis_path
        2. Dataset-specific CSV files in analysis_path/{dataset_name}/
        """
        print("💾 Saving difficulty breakdown...")
        
        filename = f"difficulty_breakdown_{self.checkpoint_days}days.csv"
        
        # Get all metrics used in difficulty calculation
        metric_weights = self.difficulty_rating_config.get("metric_weights", {})
        
        headers = ["Dataset", "Camera Name", "Overall Difficulty Score", "Difficulty Level"]
        
        # Add headers for each metric component
        for metric_name in metric_weights.keys():
            headers.extend([
                f"{metric_name}_raw",
                f"{metric_name}_normalized", 
                f"{metric_name}_weight",
                f"{metric_name}_contribution"
            ])
        
        # Collect all data first
        all_data = []
        dataset_data = defaultdict(list)
        
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "ckp" not in camera_data:
                    continue
                
                # Calculate difficulty rating with component breakdown
                difficulty_score, difficulty_level, component_scores = self._calculate_difficulty_rating(camera_data)
                
                row = {
                    "Dataset": dataset_name,
                    "Camera Name": camera_name,
                    "Overall Difficulty Score": round(difficulty_score, 4),
                    "Difficulty Level": difficulty_level,
                }
                
                # Add component scores
                for metric_name, scores in component_scores.items():
                    row[f"{metric_name}_raw"] = round(scores['raw_value'], 4)
                    row[f"{metric_name}_normalized"] = round(scores['normalized_value'], 4)
                    row[f"{metric_name}_weight"] = scores['weight']
                    row[f"{metric_name}_contribution"] = round(scores['contribution'], 4)
                
                all_data.append(row)
                dataset_data[dataset_name].append(row)

        # Save global CSV file (all datasets)
        global_output_path = os.path.join(self.analysis_path, filename)
        self._write_csv_file(global_output_path, headers, all_data)
        print(f"✅ Global difficulty breakdown saved to: {global_output_path}")

        # Save dataset-specific CSV files
        for dataset_name, dataset_rows in dataset_data.items():
            dataset_dir = os.path.join(self.analysis_path, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset_output_path = os.path.join(dataset_dir, filename)
            self._write_csv_file(dataset_output_path, headers, dataset_rows)
            print(f"✅ Dataset '{dataset_name}' difficulty breakdown saved to: {dataset_output_path}")