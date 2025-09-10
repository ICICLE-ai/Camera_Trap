from tools.pipeline import run

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run analysis and prepare images on datasets and cameras.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--days", type=int, nargs='+', default=[10, 15, 30, 45, 60], 
                        help="List of day intervals to test (default: [10, 15, 30, 45, 60])")
    parser.add_argument("--run_analysis", action='store_true', help="Run the analysis (default: False)")
    parser.add_argument("--prepare_data", action='store_true', help="Prepare JSON data files (default: False)")
    parser.add_argument("--crop_images", action='store_true', help="Crop images (only needed once per camera) (default: False)")
    parser.add_argument("--dataset", type=str, nargs='+', default=None, 
                        help="List of datasets to override the ones in config file (default: None, uses config file)")
    parser.add_argument("--wandb", action='store_true', help="Enable Weights & Biases logging (default: False)")
    
    args = parser.parse_args()

    # Run the analysis with multiple day intervals
    run(args.config, args.days, run_analysis=args.run_analysis, prepare_data=args.prepare_data, crop_images_once=args.crop_images, datasets=args.dataset, wandb_enabled=args.wandb)