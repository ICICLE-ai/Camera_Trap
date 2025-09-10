"""
Utility loaders for camera trap data processing.
Combines settings and taxonomy loading functionality.
"""

import os
import csv
import yaml
from typing import Dict, List, Optional, Tuple


class SettingsLoader:
    """Configuration loader for camera trap data processing settings."""
    
    def __init__(self, settings_path: str = "datasets/setting.yaml"):
        """
        Initialize the settings loader.
        
        Args:
            settings_path (str): Path to the settings YAML file
        """
        self.settings_path = settings_path
        self.settings = {}
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from the YAML file."""
        if not os.path.exists(self.settings_path):
            print(f"⚠️  Warning: Settings file not found: {self.settings_path}")
            return
        
        try:
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                self.settings = yaml.safe_load(f)
            print(f"✅ Loaded settings from {self.settings_path}")
        except yaml.YAMLError as e:
            print(f"❌ Error loading settings from {self.settings_path}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error loading settings: {e}")
    
    def get_datetime_formats(self) -> List[str]:
        """
        Get the list of supported datetime input formats.
        
        Returns:
            List of datetime format strings
        """
        return self.settings.get('datetime', {}).get('input_formats', [])
    
    def get_target_datetime_format(self) -> str:
        """
        Get the target datetime format for normalization.
        
        Returns:
            Target datetime format string
        """
        return self.settings.get('datetime', {}).get('target_format', "%Y:%m:%d %H:%M:%S")
    
    def get_dataset_mappings(self) -> Dict[str, str]:
        """
        Get the dataset name mappings for taxonomy.
        
        Returns:
            Dictionary mapping pipeline names to taxonomy CSV names
        """
        return self.settings.get('taxonomy', {}).get('dataset_mappings', {})
    
    def get_setting(self, key_path: str, default=None):
        """
        Get a specific setting using dot notation.
        
        Args:
            key_path (str): Dot-separated path to the setting (e.g., 'datetime.target_format')
            default: Default value if setting not found
            
        Returns:
            The setting value or default
        """
        keys = key_path.split('.')
        current = self.settings
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def reload_settings(self):
        """Reload settings from the file."""
        self._load_settings()


class TaxonomyLoader:
    """Taxonomy integration module for camera trap data processing."""
    
    def __init__(self, taxonomy_csv_path: str = "datasets/taxonomy.csv"):
        """
        Initialize the taxonomy loader.
        
        Args:
            taxonomy_csv_path (str): Path to the taxonomy CSV file
        """
        self.taxonomy_csv_path = taxonomy_csv_path
        self.taxonomy_data = {}
        self.dataset_mapping = self._load_dataset_mapping()
        self._load_taxonomy_data()
    
    def _load_dataset_mapping(self) -> Dict[str, str]:
        """
        Load dataset mapping from settings configuration.
        
        Returns:
            Dict mapping pipeline names to taxonomy CSV names
        """
        settings = get_settings_loader()
        mapping = settings.get_dataset_mappings()
        
        if not mapping:
            print("⚠️  Warning: No dataset mappings found in settings.yaml, using fallback")
            # Fallback mapping in case settings file is not available
            mapping = {
                "APN": "Snapshot Safari 2024 Expansion",
                "CDB": "Snapshot Safari 2024 Expansion",
                "ENO": "Snapshot Safari 2024 Expansion",
                "serengeti": "Snapshot Serengeti",
                "na": "NACTI",
                "orinoquia": "Orinoquia Camera Traps",
                "idaho": "Idaho Camera Traps",
                "wcs": "WCS Camera Traps",
                "island_conservation": "Island Conservation Camera Traps",
                "nz": "Trail Camera Images of New Zealand Animals"
            }
        
        return mapping
    
    def _load_taxonomy_data(self):
        """Load taxonomy data from the CSV file into memory."""
        if not os.path.exists(self.taxonomy_csv_path):
            print(f"⚠️  Warning: Taxonomy file not found: {self.taxonomy_csv_path}")
            return
        
        print(f"📖 Loading taxonomy data from {self.taxonomy_csv_path}")
        
        with open(self.taxonomy_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_name = row['dataset_name']
                query = row['query']
                
                if dataset_name not in self.taxonomy_data:
                    self.taxonomy_data[dataset_name] = {}
                
                self.taxonomy_data[dataset_name][query] = {
                    'scientific_name': row['scientific_name'],
                    'common_name': row['common_name'],
                    'taxonomy_level': row['taxonomy_level']
                }
        
        print(f"✅ Loaded taxonomy data for {len(self.taxonomy_data)} datasets")
        
        # Print dataset mapping information
        print("\n📋 Dataset name mappings:")
        for pipeline_name, csv_name in self.dataset_mapping.items():
            count = len(self.taxonomy_data.get(csv_name, {}))
            print(f"  {pipeline_name} -> {csv_name} ({count} species)")
    
    def get_taxonomy_info(self, pipeline_dataset_name: str, query: str) -> Optional[Tuple[str, str]]:
        """
        Get taxonomy information for a query from a specific dataset.
        
        Args:
            pipeline_dataset_name (str): Dataset name as used in the pipeline (e.g., 'safari')
            query (str): The query/class name to look up
            
        Returns:
            Tuple of (scientific_name, common_name) or None if not found
        """
        # Map pipeline dataset name to taxonomy CSV dataset name
        csv_dataset_name = self.dataset_mapping.get(pipeline_dataset_name)
        
        if not csv_dataset_name:
            return None
        
        # Look up the taxonomy info
        dataset_taxonomy = self.taxonomy_data.get(csv_dataset_name, {})
        taxonomy_info = dataset_taxonomy.get(query)
        
        if taxonomy_info:
            return (
                taxonomy_info['scientific_name'], 
                taxonomy_info['common_name']
            )
        
        return None
    
    def get_available_datasets(self) -> Dict[str, str]:
        """
        Get the available dataset mappings.
        
        Returns:
            Dictionary of pipeline_name -> csv_name mappings
        """
        return self.dataset_mapping.copy()
    
    def print_dataset_mappings(self):
        """Print the dataset name mappings that were applied."""
        print("\n🔄 Dataset name mappings applied:")
        for pipeline_name, csv_name in self.dataset_mapping.items():
            if csv_name in self.taxonomy_data:
                species_count = len(self.taxonomy_data[csv_name])
                print(f"  ✅ {pipeline_name} -> {csv_name} ({species_count} species)")
            else:
                print(f"  ❌ {pipeline_name} -> {csv_name} (not found in taxonomy CSV)")


# Global loader instances
_settings_loader = None
_taxonomy_loader = None


def get_settings_loader() -> SettingsLoader:
    """Get the global settings loader instance."""
    global _settings_loader
    if _settings_loader is None:
        _settings_loader = SettingsLoader()
    return _settings_loader


def get_taxonomy_loader() -> TaxonomyLoader:
    """Get the global taxonomy loader instance."""
    global _taxonomy_loader
    if _taxonomy_loader is None:
        _taxonomy_loader = TaxonomyLoader()
    return _taxonomy_loader
