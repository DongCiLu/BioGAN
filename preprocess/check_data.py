from shutil import copyfile
import os
from PIL import Image

src_dir = "classification/unsupervised"

cnt = 0
error_cnt = 0
for subdir, dirs, files in os.walk(src_dir):
    for filename in files:
        src_fn = os.path.join(subdir, filename)
        im = Image.open(src_fn)
        if im.size != (128, 128):
            error_cnt += 1
            print src_fn
        cnt += 1

print ("{} errors out of {}".format(error_cnt, cnt))
