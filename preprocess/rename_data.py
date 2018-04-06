from shutil import copyfile
import os
from PIL import Image

src_dir = 'classification/unsupervised'
dst_dir = 'classification/unsupervised_combined'

cnt = 0
ctn_cnt = 0
for subdir, dirs, files in os.walk(src_dir):
    for fn in files:
        cnt += 1
        emb_no = subdir[-1]
        emb_index = fn.find('emb')
        new_fn = fn[:emb_index + 3] + emb_no + fn[emb_index + 4:]
        src_fn = os.path.join(subdir, fn)
        dst_subdir = subdir[:27] + '_combined' 
        dst_fn = os.path.join(dst_subdir, new_fn)
        copyfile(src_fn, dst_fn)

