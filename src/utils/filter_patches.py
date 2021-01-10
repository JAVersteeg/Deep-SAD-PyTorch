import click
import numpy as np
from PIL import Image
import sys
import time
import os
from shutil import copyfile
import matplotlib.image as mpimg

# for 128x128 patches
# python filter_patches.py ../../../patches_1119/crack/binarylabel ../../../patches_1119/crack/img ../../../patches_1119/crack/crack_ncc ../../../patches_1119/crack/crack_ncc_remaining --patch_size 128


@click.command()

@click.argument('bin_path', type=click.Path(exists=True))
@click.argument('crack_path', type=click.Path(exists=True))
@click.argument('new_crack_path', type=click.Path(exists=True))
@click.argument('remaining_crack_path', type=click.Path(exists=True))
@click.option('--patch_size', type=int, default=64, help='The size of a crack patch, is always a square (N x N)')

def copy_non_corner_crack_files(bin_path, crack_path, new_crack_path, remaining_crack_path, patch_size: int = 64):
    print("Filtering", os.path.join(crack_path), "into", new_crack_path, "based on", os.path.join(bin_path))
    
    for filename in os.listdir(bin_path):
        crack_folder_filepath = os.path.join(crack_path, filename)
        filtered_crack_folder_filepath = os.path.join(new_crack_path, filename)
        remaining_crack_folder_filepath = os.path.join(remaining_crack_path, filename)
        
        img = mpimg.imread(os.path.join(bin_path, filename))
        
        if not has_corner_crack(img, patch_size) and not "inverted" in filename:
            copyfile(crack_folder_filepath, filtered_crack_folder_filepath)
        elif not "inverted" in filename:
            copyfile(crack_folder_filepath, remaining_crack_folder_filepath)
        else:
            pass

def has_corner_crack(x: np.array, patch_size: int = 64):
    """ As long as there are no non-black pixels in the center of the patch, continue
        If there are, this patch does not contain a corner crack."""
        
    if patch_size == 64:
        corner_range = range(16, 48)
    else:
        corner_range = range(32, 96)
    pixel_found = False  # is there a non-corner crack pixel?
    normal_cracks = np.array([])
    for i, row in enumerate(x):
        if i in corner_range and not pixel_found:       
            for j, pixel in enumerate(row):
                if j in corner_range and not pixel_found:       
                    if pixel > 0.0:
                        pixel_found = True
                        return not pixel_found
    return not pixel_found
    
def remove_dataset2():
    folder_path = ("../../../patches_1119/crack/img_ncc")
    
    for filename in os.listdir(folder_path):
        if "Dataset2" in filename:
            os.remove(os.path.join(folder_path, filename))
    
def remove_inverted():
    folder_path = ("../../../patches_1119/crack/img_no_inv")
    
    for filename in os.listdir(folder_path):
        if "inverted" in filename:
            os.remove(os.path.join(folder_path, filename))

if __name__ == '__main__':
    remove_dataset2()