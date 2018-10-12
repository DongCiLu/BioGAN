import os
import math
import random 
import argparse
from shutil import copyfile

BASE_RATIO = 0.66667 # train percentage in full dataset

def moving_data(src_dir, dst_dir, ros_class, num_train, num_test):
    train_path = os.path.join(dst_dir, "train")
    test_path = os.path.join(dst_dir, "test")
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    train_rc_path = os.path.join(train_path, "{}".format(ros_class))
    test_rc_path = os.path.join(test_path, "{}".format(ros_class))
    if not os.path.exists(train_rc_path):
        os.mkdir(train_rc_path)
    if not os.path.exists(test_rc_path):
        os.mkdir(test_rc_path)
    random.seed() # use current time as the seed
    fn_list = []
    for subdir, dirs, files in os.walk(src_dir):
        for filename in files:
            src_fn = os.path.join(subdir, filename)
            fn_list.append((filename, src_fn))

    remaining_fn = len(fn_list)
    num_train_fullfilled = 0
    while num_train_fullfilled < num_train :
        r = random.randint(0, remaining_fn - 1)
        target_fn = os.path.join(dst_dir, 'train', ros_class, fn_list[r][0])
        copyfile(fn_list[r][1], target_fn)
        fn_list[r], fn_list[remaining_fn - 1] = \
                fn_list[remaining_fn - 1], fn_list[r] 
        remaining_fn -= 1
        num_train_fullfilled += 1

    num_test_fullfilled = 0
    while num_test_fullfilled < num_test :
        r = random.randint(0, remaining_fn - 1)
        target_fn = os.path.join(dst_dir, 'test', ros_class, fn_list[r][0])
        copyfile(fn_list[r][1], target_fn)
        fn_list[r], fn_list[remaining_fn - 1] = \
                fn_list[remaining_fn - 1], fn_list[r] 
        remaining_fn -= 1
        num_test_fullfilled += 1
    print (num_test_fullfilled, num_test)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('network_size', type=int)
    ap.add_argument('ros_nros_ratio', type=str)
    ap.add_argument('train_ratio', type=float)
    args = ap.parse_args()

    # TODO: do it more clever later, no motivation atm
    if args.network_size == 128:
        ros_src_dir = "datasets/classification/supervised_128/data/ros"
        nros_src_dir = "datasets/classification/supervised_128/data/nros"
        dst_dir = "datasets/classification/supervised_128"
        ros_total = 386
        nros_total = 1192
    elif args.network_size == 32:
        ros_src_dir = "datasets/classification/supervised_32/data/ros"
        nros_src_dir = "datasets/classification/supervised_32/data/nros"
        dst_dir = "datasets/classification/supervised_32"
        ros_total = 78
        nros_total = 211

    if args.ros_nros_ratio == '1to3':
        ros_percentage = 1.0
        nros_percentage = 1.0
    elif args.ros_nros_ratio == '1to1':
        ros_percentage = 1.0
        nros_percentage = 1.0 / 3.0
    
    num_ros = ros_total * ros_percentage
    num_nros = nros_total * nros_percentage

    num_train_ros = \
            int(round(num_ros * BASE_RATIO * args.train_ratio))
    num_test_ros = int(round(num_ros * (1 - BASE_RATIO)))
    num_train_nros = \
            int(round(num_nros * BASE_RATIO * args.train_ratio))
    num_test_nros = int(round(num_nros * (1 - BASE_RATIO)))

    print("*** {}, {}".format(num_train_ros, num_test_ros))
    moving_data(ros_src_dir, dst_dir, "rosette", num_train_ros, num_test_ros)
    print("*** {}, {}".format(num_train_nros, num_test_nros))
    moving_data(nros_src_dir, dst_dir, "non_rosette", num_train_nros, num_test_nros)
    print("*** {}, {}".format(num_train_nros + num_train_ros, num_test_nros + num_test_ros))
