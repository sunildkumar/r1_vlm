#!/usr/bin/env python3
"""
Script to download the MNIST dataset and save images in an obfuscated format.
Each image is resized to 224x224 pixels and saved as a JPEG with a randomized filename.
A corresponding JSON file with the correct label is saved next to it.
Data is stored in the ./data directory.
"""

import json
import random
import string
from pathlib import Path

from PIL import Image
from torchvision.datasets import MNIST


def get_unique_filename(data_dir: Path, length: int = 5) -> str:
    """
    Generate a unique random filename base in the given directory.
    
    Returns:
        A random string of the specified length that does not conflict with existing files.
    """
    while True:
        candidate = "".join(random.choices(string.ascii_lowercase, k=length))
        image_file = data_dir / f"img-{candidate}.jpeg"
        json_file = data_dir / f"img-{candidate}.json"
        if not image_file.exists() and not json_file.exists():
            return candidate


def main():
    """
    Download the MNIST training dataset and save each image and its label in obfuscated form.
    
    Each image is resized to 224x224 pixels and saved as a JPEG with a randomized filename.
    A corresponding JSON file with the label is saved next to each image.
    """
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True, parents=True)

    print("Downloading MNIST dataset (training set)...")
    mnist_dataset = MNIST(root="./mnist_data", train=True, download=True)

    total_images = len(mnist_dataset)
    print(f"Processing {total_images} images...")

    for index in range(total_images):
        image, label = mnist_dataset[index]

        # Resize image to 224x224 pixels using a high-quality resampling filter.
        resized_image = image.resize((224, 224), resample=Image.LANCZOS)

        # Generate a unique randomized filename base.
        unique_str = get_unique_filename(data_dir)

        image_filename = data_dir / f"img-{unique_str}.jpeg"
        json_filename = data_dir / f"img-{unique_str}.json"

        # Save the JPEG image.
        resized_image.save(str(image_filename), format="JPEG")

        # Save the label to a JSON file.
        with open(json_filename, "w") as json_file_handle:
            json.dump({"label": label}, json_file_handle)

        if (index + 1) % 1000 == 0 or (index + 1) == total_images:
            print(f"Processed {index + 1}/{total_images} images.")

    print("Done.")


if __name__ == "__main__":
    main() 