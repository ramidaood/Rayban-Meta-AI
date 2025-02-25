import os
import sys

# 1) Import the relevant functions from openimages
try:
    from openimages.download import download_images, download_annotations
except ImportError:
    print("Please install openimages:\n  pip install openimages")
    sys.exit(1)

def main():
    # Classes you want from Open Images
    classes = [
        'Car',
        'Bus',
        'Motorcycle',
        'Bicycle',
        'Trash can',
        'Traffic cone',
        'Fountain',
        'Bench',
        'Pole',
        'Fire hydrant',
        'Mailbox',
        'Chair',
        'Table',
        'Door',
        'Sign',
        'Bookshelf',
        'Big table',  # May not exist in Open Images
        'Wall'
    ]

    # Directory to store downloaded data
    base_dir = 'openimages_data'
    images_dir = os.path.join(base_dir, 'train')
    os.makedirs(images_dir, exist_ok=True)

    # Download images
    print("=== Downloading Images ===")
    download_images(
        labels=classes,
        image_limit=500,  # up to 500 images per label (adjust as needed)
        dataset_dir=images_dir,
        annotation_format=None  # We'll download annotations in the next step
    )

    # Download bounding box annotations
    # We'll try 'yolo' format. If your openimages version doesn't support it, switch to 'pascal' or 'csv'.
    print("=== Downloading Annotations ===")
    try:
        download_annotations(
            labels=classes,
            dataset_dir=images_dir,
            annotation_format='yolo'
        )
        print("Annotations downloaded in YOLO format.")
    except Exception as e:
        print(f"YOLO annotation download failed with error:\n{e}")
        print("Attempting 'pascal' format...")
        download_annotations(
            labels=classes,
            dataset_dir=images_dir,
            annotation_format='pascal'
        )
        print("Annotations downloaded in PASCAL format. You'll need to convert to YOLO if required.")

if __name__ == '__main__':
    main()