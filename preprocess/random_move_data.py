from shutil import copyfile
from random import random
import os

src_dir = "classification/all_rosettes"
# src_dir = "batch3_processed/1non"
dst_dir = "classification/celegans_aros_dataset"

for subdir, dirs, files in os.walk(src_dir):
    for filename in files:
        src_fn = os.path.join(subdir, filename)
        
        r = random()

        if r > 0.66667:
            target = '/test/rosette'
            # target = '/test/non_rosette'
        else:
            target = '/train/rosette'
            # target = '/train/non_rosette'

        target_dir = dst_dir + target
        target_fn = os.path.join(target_dir, filename)
        copyfile(src_fn, target_fn)
