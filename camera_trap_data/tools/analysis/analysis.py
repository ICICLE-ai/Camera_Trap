import yaml
import os
from tools.analysis.metric_analysis import MetricAnalysis
from tools.analysis.plot_analysis import PlotAnalysis

# Try to import settings loader, fallback if not available
try:
    from tools.data_prep.utils.loader_utils import get_settings_loader
    _settings_available = True
except ImportError:
    _settings_available = False

class Analysis:
    def __init__(self, config_path, dataset, datetime_formats=None, checkpoint_days=30, config=None):
        """
        Initialize the AnalysisManager, load settings, and execute analysis.

        Args:
            config_path (str): Path to the YAML configuration file.
            dataset (Dataset): Prepared dataset object.
            datetime_formats (list): List of datetime format strings to try when parsing dates.
            checkpoint_days (int): Number of days for each checkpoint interval.
            config (dict): Optional pre-loaded configuration dictionary. If None, loads from config_path.
        """
        self.config_path = config_path
        self.dataset = dataset
        
        # Load datetime formats from settings if available, otherwise use provided or default
        if datetime_formats is None and _settings_available:
            try:
                settings = get_settings_loader()
                self.datetime_formats = settings.get_datetime_formats()
            except Exception:
                self.datetime_formats = ["%Y:%m:%d %H:%M:%S"]  # Default fallback
        else:
            self.datetime_formats = datetime_formats or ["%Y:%m:%d %H:%M:%S"]  # Default fallback
        
        self.checkpoint_days = checkpoint_days
        self.config = config if config is not None else self._load_config()
        self.analysis_path = self.config.get("analysis_path", "analysis_results")

        # Set up directory structure
        self._setup_directories()

        # Initialize metric and plot analysis modules
        self.metric_analysis = MetricAnalysis(self.config, self.dataset, self.analysis_path, self.datetime_formats, self.checkpoint_days)
        self.plot_analysis = PlotAnalysis(self.config, self.dataset, self.analysis_path, self.checkpoint_days)

        # Execute analysis immediately
        print(f"\n======================================== Running Analysis ({self.checkpoint_days} days) =========================================\n")
        self.run_all_analyses()
        self.generate_report()
        print()

    def _load_config(self):
        """Load the YAML configuration file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def _setup_directories(self):
        """Set up the directory structure for saving analysis results."""
        for dataset_name, cameras in self.dataset.metadata.items():
            dataset_path = os.path.join(self.analysis_path, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            for camera_name in cameras.keys():
                # Create day-specific directories for plots
                camera_path = os.path.join(dataset_path, camera_name, str(self.checkpoint_days), "plots", "ckp_piechart")
                os.makedirs(camera_path, exist_ok=True)
                histogram_path = os.path.join(dataset_path, camera_name, str(self.checkpoint_days), "plots", "histograms")
                os.makedirs(histogram_path, exist_ok=True)

    def run_all_analyses(self):
        """
        Run all analyses (metrics and plots) based on the configuration.
        """
        self.metric_analysis.run()
        self.plot_analysis.run()

    def generate_report(self):
        """
        Generate and save the analysis report.
        """
        print("\n======================================== Generating Report =========================================\n")
        # Example: Print or save results
        for dataset_name, cameras in self.dataset.metadata.items():
            for camera_name, camera_data in cameras.items():
                if "analysis" in camera_data:
                    print(f"Dataset: {dataset_name}, Camera: {camera_name}")
                    print(camera_data["analysis"])