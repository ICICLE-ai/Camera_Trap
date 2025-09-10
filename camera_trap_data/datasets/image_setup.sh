#!/bin/bash

# Script to link dataset images with their actual locations
# This script creates symbolic links from dataset directories to actual image locations

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Dictionary of dataset names and their corresponding image paths
# Paths based on actual folder names in /fs/scratch/PAS2099/camera-trap/
declare -A DATASET_PATHS=(
    ["na"]="/fs/scratch/PAS2099/camera-trap/north-amerian-camera-trap-images"
    # ["caltech"]="/fs/scratch/PAS2099/camera-trap/caltech"
    ["channel_island"]="/fs/scratch/PAS2099/camera-trap/channel_island"
    ["idaho"]="/fs/scratch/PAS2099/camera-trap/idaho-camera-trap"
    ["island_conservation"]="/fs/scratch/PAS2099/camera-trap/island_conservation"
    ["missouri"]="/fs/scratch/PAS2099/camera-trap/missouri"
    ["nz"]="/fs/scratch/PAS2099/camera-trap/new-zealand"
    ["orinoquia"]="/fs/scratch/PAS2099/camera-trap/orinoquia"
    ["seattle"]="/fs/scratch/PAS2099/camera-trap/seattle"
    ["serengeti"]="/fs/scratch/PAS2099/camera-trap/serengeti"
    ["swg"]="/fs/scratch/PAS2099/camera-trap/swg"
    ["wcs"]="/fs/scratch/PAS2099/camera-trap/wcs"
    # ["wellington"]="/fs/scratch/PAS2099/camera-trap/wellington"
)

# Function to create symbolic link for a dataset
create_dataset_link() {
    local dataset_name=$1
    local image_path=$2
    
    print_status "Processing dataset: $dataset_name"
    
    # Check if dataset directory exists
    if [ ! -d "$SCRIPT_DIR/$dataset_name" ]; then
        print_error "Dataset directory $SCRIPT_DIR/$dataset_name does not exist"
        return 1
    fi
    
    # Create images directory if it doesn't exist
    local images_dir="$SCRIPT_DIR/$dataset_name/images"
    if [ ! -d "$images_dir" ]; then
        print_status "Creating images directory: $images_dir"
        mkdir -p "$images_dir"
    fi
    
    # Change to the images directory
    cd "$images_dir" || {
        print_error "Failed to change to directory: $images_dir"
        return 1
    }
    
    # Check if source path exists
    if [ ! -d "$image_path" ]; then
        print_error "Source image path does not exist: $image_path"
        return 1
    fi
    
    # Remove existing symbolic link if it exists
    if [ -L "." ] || [ -L "$(basename "$image_path")" ]; then
        print_warning "Removing existing symbolic link in $images_dir"
        find . -maxdepth 1 -type l -delete
    fi
    
    # Create symbolic links efficiently using find with xargs
    print_status "Creating symbolic links: $image_path/* -> $images_dir"
    
    # Count total items for progress indication
    local total_items=$(find "$image_path" -mindepth 1 -maxdepth 1 | wc -l)
    print_status "Found $total_items items to link"
    
    # Use find with xargs for much better performance
    find "$image_path" -mindepth 1 -maxdepth 1 -print0 | xargs -0 -I {} ln -s {} . || {
        print_error "Failed to create symbolic links for $dataset_name"
        return 1
    }
    
    print_status "Successfully linked $dataset_name to $image_path"
    return 0
}

# Main execution
main() {
    print_status "Starting image setup script..."
    print_status "Script directory: $SCRIPT_DIR"
    
    local success_count=0
    local error_count=0
    
    # Process each dataset
    for dataset_name in "${!DATASET_PATHS[@]}"; do
        image_path="${DATASET_PATHS[$dataset_name]}"
        
        if create_dataset_link "$dataset_name" "$image_path"; then
            ((success_count++))
        else
            ((error_count++))
        fi
        
        echo # Empty line for readability
    done
    
    # Summary
    print_status "Setup completed!"
    print_status "Successfully processed: $success_count datasets"
    if [ $error_count -gt 0 ]; then
        print_error "Failed to process: $error_count datasets"
        exit 1
    else
        print_status "All datasets processed successfully!"
    fi
}

# Run the main function
main "$@"