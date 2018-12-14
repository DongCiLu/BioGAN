# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the MNIST dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_mnist.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim

_DATASET_CONFIG = ""
_FILE_PATTERN = ""
_SPLITS_TO_SIZES = {'unlabeled': 0, 'train': 0, 'test': 0, 'predict': 0}
_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [256 x 256 x 1] grayscale image.',
    'label': 'A single integer between 08 and 19',
}

def config_dataset(data_config):
    segs = data_config.split('_')
    network_size = segs[0]
    ros_ratio = segs[1]
    train_ratio = segs[2]
    dataset_no = segs[3]
    train_size = segs[4]
    test_size = segs[5]
    if network_size == "128":
        unlabeled_size = 11250
    if network_size == "64":
        unlabeled_size = 45000
    elif network_size == "32":
        unlabeled_size = 180000
    _FILE_PATTERN = "celegans-%s" + "_{}_{}_{}.tfrecord".format(
            ros_ratio, train_ratio, dataset_no)
    _SPLITS_TO_SIZES = {'unlabeled': unlabeled_size, 
                        'train': train_size, 
                        'test': test_size, 
                        'predict': test_size}

    print ("---------------".format(_FILE_PATTERN))
    return _FILE_PATTERN, _SPLITS_TO_SIZES

def get_split(split_name, dataset_dir, 
        file_pattern=None, reader=None, mode="", data_config=""):
    """Gets a dataset tuple with instructions for reading MNIST.
  
    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.
  
    Returns:
      A `Dataset` namedtuple.
  
    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """

    _FILE_PATTERN, _SPLITS_TO_SIZES = config_dataset(data_config)
  
    if split_name not in _SPLITS_TO_SIZES:
      raise ValueError('split name %s was not recognized.' % split_name)
  
    if not file_pattern:
      file_pattern = _FILE_PATTERN
    print("Reading dataset with file pattern: {}".format(_FILE_PATTERN))
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
  
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
      reader = tf.TFRecordReader
  
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature(
            [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value='no_filename'),
        # 'image/filename': tf.VarLenFeature(dtype=tf.string),
    }
  
    base_size = 128
    if mode == "multiple":
      width = base_size * 2
      height = base_size
    elif mode == "medium" or mode == "mediumgan":
      width = 64
      height = 64
    elif mode == "tiny" or mode == "tinygan":
      width = 32
      height = 32
    else:
      width = base_size
      height = base_size
  
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[height, width, 1], channels=1),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
    }
  
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
  
    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
      labels_to_names = dataset_utils.read_label_file(dataset_dir)
  
    return slim.dataset.Dataset(
        data_sources=file_pattern, 
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        num_classes=_NUM_CLASSES,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        labels_to_names=labels_to_names)
