import os
import sys
import math
import random 
import argparse
from shutil import copyfile

def moving_data(src_dir, dst_dir, ros_class, num_train):
    train_path = os.path.join(dst_dir, "train")
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    train_rc_path = os.path.join(train_path, "{}".format(ros_class))
    if not os.path.exists(train_rc_path):
        os.mkdir(train_rc_path)
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('network_size', type=int)
    ap.add_argument('ros_nros_ratio', type=str)
    ap.add_argument('train_dir', type=str)
    ap.add_argument('train_ratio', type=float)
    args = ap.parse_args()

    # only support 32 network atm
    if args.network_size == 32:
        base_dir = "datasets/classification/supervised_32"
        ros_src_dir = "{}/{}/ros".format(base_dir, args.train_dir)
        nros_src_dir = "{}/{}/nros".format(base_dir, args.train_dir)
        dst_dir = base_dir

    if args.train_dir == 'train1':
        if args.ros_nros_ratio == "1to3":
            ros_total = 51
            nros_total = 142
        elif args.ros_nros_ratio == "1to1":
            ros_total = 51
            nros_total = 51
    elif args.train_dir == 'train2':
        if args.ros_nros_ratio == "1to3":
            ros_total = 58
            nros_total = 149
        elif args.ros_nros_ratio == "1to1":
            ros_total = 58
            nros_total = 58
    elif args.train_dir == 'train3':
        if args.ros_nros_ratio == "1to3":
            ros_total = 47
            nros_total = 131
        elif args.ros_nros_ratio == "1to1":
            ros_total = 47
            nros_total = 47

    num_train_ros = int(round(ros_total * args.train_ratio))
    num_train_nros = int(round(nros_total * args.train_ratio))

    moving_data(ros_src_dir, dst_dir, "rosette", num_train_ros)
    moving_data(nros_src_dir, dst_dir, "non_rosette", num_train_nros)
    print("trainset stat: {} total, {} rosette, {} non-rosette".format(
        num_train_nros + num_train_ros, num_train_ros, num_train_nros))
