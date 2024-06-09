import argparse
from PIL import Image
import os

def resize_images_in_directory(directory: str, size: tuple = (128, 128)):
    # List all files in the directory
    files = os.listdir(directory)

    # Iterate through each file
    for file in files:
        # Check if the file is a PNG image
        if file.endswith('.png'):
            # Open the image file
            with Image.open(os.path.join(directory, file)) as img:
                # Resize the image to the specified size
                img_resized = img.resize(size)
                # Save the resized image back to the same location
                img_resized.save(os.path.join(directory, file))

    # Return the list of files to confirm changes
    return os.listdir(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all PNG images in a directory to 128x128.")
    parser.add_argument("directory", type=str, help="The directory containing the PNG images to resize.")
    args = parser.parse_args()

    resized_files = resize_images_in_directory(args.directory)
    print(resized_files)

