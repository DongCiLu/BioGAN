from shutil import copyfile
from random import random
import os

ros_src_dir = "../../Datasets/classification/supervised/all/rosette_combined"
non_ros_src_dir = "../../Datasets/classification/supervised/all/non_rosette"
dst_dir = "../../Datasets/classification/supervised"

base_ratio = 0.66667

def moving_data(src_dir, ros_class, percentage=1.0):
    if not os.path.exists(os.path.join(dst_dir, "train_{}".format(percentage))):
        os.mkdir(os.path.join(dst_dir, "train_{}".format(percentage)))
    if not os.path.exists(os.path.join(dst_dir, "test_{}".format(percentage))):
        os.mkdir(os.path.join(dst_dir, "test_{}".format(percentage)))
    os.mkdir(os.path.join(dst_dir, "train_{}/{}".format(percentage, ros_class)))
    os.mkdir(os.path.join(dst_dir, "test_{}/{}".format(percentage, ros_class)))
    for subdir, dirs, files in os.walk(src_dir):
        for filename in files:
            src_fn = os.path.join(subdir, filename)
            r = random()
            if r > base_ratio + (1 - base_ratio) * (1 - percentage):
                target = '/test_{}/{}'.format(percentage, ros_class)
            elif r < base_ratio * percentage:
                target = '/train_{}/{}'.format(percentage, ros_class)
            else:
                continue
            target_dir = dst_dir + target
            target_fn = os.path.join(target_dir, filename)
            copyfile(src_fn, target_fn)

ros_percentage = 0.9999 
non_ros_percentage = 0.3333
moving_data(ros_src_dir, "rosette", ros_percentage)
moving_data(non_ros_src_dir, "non_rosette", non_ros_percentage)
