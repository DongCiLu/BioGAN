from PIL import Image
import numpy as np

import os

src_dir = "celegans"
dst_dir = "celegans_multi_cond"
time_interval = 8

# assume the file is read in order

dst_dir += "_{}".format(time_interval)
for subdir, dirs, files in os.walk(src_dir):
    for fn in files:
        if len(subdir) < 9:
            continue
        src_fn = os.path.join(subdir, fn)

        slice_beg = fn.find("_t") + 2
        slice_end = fn.find(".jpg")
        time_slice = int(fn[slice_beg:slice_end])
        next_fn = fn[0:slice_beg] + \
                "{}".format(time_slice + time_interval) + \
                fn[slice_end:]
        next_src_fn = os.path.join(subdir, next_fn)

        if os.path.isfile(next_src_fn):
            images_list = [src_fn, next_src_fn]
            images = [Image.open(i) for i in images_list]

            merged_image_array = \
                    np.hstack((np.asarray(image) for image in images))
            merged_image = Image.fromarray(merged_image_array)
            new_fn = subdir[-2:] + "_" + fn
            fn_time_slice = (time_slice - 1) / 10
            dst_fn = "{}/{}/{}".format(dst_dir, fn_time_slice, new_fn)

            merged_image.save(dst_fn)

