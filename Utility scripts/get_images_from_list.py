'''
This script receives a list of image paths as a txt file, their base directory and a name for the destination directory.
It copies the images in the txt from the data directory to a directory in the same place as the txt.
This is used to collect and view the misclassified images after testing a model.
'''

import os
import shutil
import argparse

def move_images(image_paths, data_dir, dest_dir):
    full_image_paths = [os.path.join(data_dir, image_path) for image_path in image_paths]
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for image_path in full_image_paths:
        shutil.copy(image_path, dest_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', type=str, required=True, help='Path to the txt file containing the image paths')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory of the images')
    parser.add_argument('--dest_dir_name', type=str, required=True, help='Name of the directory to move the images to. Will be created in the same directory as the image_paths txt')
    args = parser.parse_args()

    dest_dir = os.path.join(os.path.dirname(args.image_paths), args.dest_dir_name)

    with open(args.image_paths, 'r') as f:
        image_paths = f.read().splitlines()

    move_images(image_paths, args.data_dir, dest_dir)