from shutil import copyfile
import os

src_dir = "celegans_neighbor"
dst_dir = "celegans_cond"

for subdir, dirs, files in os.walk(src_dir):
    for filename in files:
        src_fn = os.path.join(subdir, filename)
        # new_fn = subdir[-2:] + "_" + filename
        new_fn = subdir[-10:] + "_" + filename[35:]

        # slice_beg = new_fn.find("_t") + 2
        # slice_end = new_fn.find(".jpg")
        # time_slice = (int(new_fn[slice_beg:slice_end]) - 1) / 10

        key = "slice"
        slice_beg = new_fn.find(key) + len(key)
        slice_end = new_fn.find(".jpg")
        slice_num = new_fn[slice_beg:slice_end]

        dst_fn = "{}/{}/{}".format(dst_dir, slice_num, new_fn)
        copyfile(src_fn, dst_fn)
