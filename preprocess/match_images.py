import numpy as np
import os
from shutil import copyfile
from PIL import Image

ref_dir = "classification/predicted_rosettes/low_prob"
src_dir = "classification/unsupervised_combined/all"
dst_dir = "classification/all_rosettes/low_prob"

miss_cnt = 0
for subdir, dirs, files in os.walk(ref_dir):
    for filename in files:
        cut_index = filename.find('.') - 2
        filename = filename[:cut_index]
        filename += ".jpg"
        src_fn = os.path.join(src_dir, filename)
        dst_fn = os.path.join(dst_dir, filename)
        if os.path.isfile(src_fn):
            copyfile(src_fn, dst_fn)
        else:
            miss_cnt += 1

print("Failed to find {} files.".format(miss_cnt))
