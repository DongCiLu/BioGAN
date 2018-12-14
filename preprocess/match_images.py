import numpy as np
import os
from shutil import copyfile
from PIL import Image

ref_dir = "/data/Datasets/classification/supervised_128/one3rd_1to3/nros"
src_dir = "/data/Datasets/classification/supervised_128/full_1to3/nros"
dst_dir = "/data/Datasets/classification/supervised_128/new_nros"

miss_cnt = 0
for subdir, dirs, files in os.walk(ref_dir):
    for filename in files:
        '''
        cut_index = filename.find('.') - 2
        filename = filename[:cut_index]
        filename += ".jpg"
        '''
        src_fn = os.path.join(src_dir, filename)
        dst_fn = os.path.join(dst_dir, filename)
        if os.path.isfile(src_fn):
            copyfile(src_fn, dst_fn)
        else:
            miss_cnt += 1

print("Failed to find {} files.".format(miss_cnt))
