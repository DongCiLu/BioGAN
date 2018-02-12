from shutil import copyfile
import os

src_dir = "celegans"
dst_dir = "celegans_cond"

for subdir, dirs, files in os.walk(src_dir):
    for filename in files:
        if len(subdir) < 9:
            continue
        src_fn = os.path.join(subdir, filename)
        new_fn = subdir[-2:] + "_" + filename

        slice_beg = new_fn.find("_t") + 2
        slice_end = new_fn.find(".jpg")
        time_slice = (int(new_fn[slice_beg:slice_end]) - 1) / 10

        dst_fn = "{}/{}/{}".format(dst_dir, time_slice, new_fn)

        copyfile(src_fn, dst_fn)
