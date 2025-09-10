import json
import os

def count_wcs_images():
    """
    Read WCS metadata.json file and count the total number of images and annotations.
    
    Returns:
        tuple: (total_images, total_annotations)
    """
    # Path to the metadata file
    metadata_path = os.path.join(os.path.dirname(__file__), "metadata", "wcs_metadata.json")
    
    try:
        # Read the JSON file
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get the images attribute and count its length
        total_images = 0
        total_annotations = 0
        
        if "images" in metadata:
            total_images = len(metadata["images"])
            print(f"Total number of images in WCS metadata: {total_images}")
        else:
            print("'images' attribute not found in metadata")
            
        if "annotations" in metadata:
            total_annotations = len(metadata["annotations"])
            print(f"Total number of annotations in WCS metadata: {total_annotations}")
        else:
            print("'annotations' attribute not found in metadata")
            
        return total_images, total_annotations
            
    except FileNotFoundError:
        print(f"Metadata file not found at: {metadata_path}")
        return 0, 0
    except json.JSONDecodeError:
        print("Error decoding JSON file")
        return 0, 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0, 0

if __name__ == "__main__":
    count_wcs_images()
