import numpy as np
import os
from PIL import Image

src_dir_path = "classification/unsupervised_combined"
dst_dir_path = "classification/unsupervised_patches"
window_size = 32

def cut_image(image):
    patches = []
    for r in range(0, image.shape[0] - window_size + 1, window_size):
        for c in range(0, image.shape[1] - window_size + 1, window_size):
            patch = image[r:r+window_size, c:c+window_size]
            patches.append(patch)
    return patches

def save_patches(patches, filename):
    prefix = filename[:-4]
    # patches_dir_path = os.path.join(dst_dir_path, prefix)
    # os.makedirs(patches_dir_path)
    for cnt, patch in enumerate(patches):
        patch_image = Image.fromarray(patch)
        patch_filename = '{}_p{}.jpg'.format(prefix, cnt)
        # patch_full_filename = os.path.join(patches_dir_path, patch_filename)
        patch_full_filename = os.path.join(dst_dir_path, patch_filename)
        patch_image.save(patch_full_filename) 

if __name__ == '__main__':
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    for subdir, dirs, files in os.walk(src_dir_path):
        for filename in files:
            src_fn = os.path.join(subdir, filename)
            print('processing file {}'.format(src_fn))
            image = Image.open(src_fn)
            image = np.array(image)
            patches = cut_image(image)
            save_patches(patches, filename)
