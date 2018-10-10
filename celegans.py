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

# _DATASET_CONFIG = "128_1to1old"
_DATASET_CONFIG = ""
_FILE_PATTERN = ""
_SPLITS_TO_SIZES = {'unlabeled': 0, 'train': 0, 'test': 0, 'predict': 0}
_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [256 x 256 x 1] grayscale image.',
    'label': 'A single integer between 08 and 19',
}

def config_dataset(data_config):
    if data_config == "128_default":
        _FILE_PATTERN = "celegans-%s.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 528, 
                            'test': 253, 
                            'predict': 11250}
    elif data_config == "128_1to1":
        _FILE_PATTERN = "celegans-%s_1to1.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 528, 
                            'test': 270, 
                            'predict': 11250}
    elif data_config == "128_1to1old":
        _FILE_PATTERN = "celegans-%s_1to1old.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 521, 
                            'test': 253, 
                            'predict': 11250}
    elif data_config == "128_1.0":
        _FILE_PATTERN = "celegans-%s_1.0.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 1063, 
                            'test': 523, 
                            'predict': 11250}
    elif data_config == "128_0.5": 
        _FILE_PATTERN = "celegans-%s_0.5.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 527, 
                            'test': 523, 
                            'predict': 11250}
    elif data_config == "128_0.2": 
        _FILE_PATTERN = "celegans-%s_0.2.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 203, 
                            'test': 523, 
                            'predict': 11250}
    elif data_config == "128_0.1": 
        _FILE_PATTERN = "celegans-%s_0.1.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 100, 
                            'test': 523, 
                            'predict': 11250}
    elif data_config == "128_0.05": 
        _FILE_PATTERN = "celegans-%s_0.05.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 53, 
                            'test': 523, 
                            'predict': 11250}
    elif data_config == "128_0.02": 
        _FILE_PATTERN = "celegans-%s_0.02.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 28, 
                            'test': 523, 
                            'predict': 11250}
    elif data_config == "128_0.01": 
        _FILE_PATTERN = "celegans-%s_0.01.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 11250, 
                            'train': 10, 
                            'test': 523, 
                            'predict': 11250}
    elif data_config == "32_1.0": 
        _FILE_PATTERN = "tiny-%s_1.0.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 180000, 
                            'train': 220, 
                            'test': 69, 
                            'predict': 69}
    elif data_config == "32_0.3333": 
        _FILE_PATTERN = "tiny-%s_0.3333.tfrecord"
        _SPLITS_TO_SIZES = {'unlabeled': 180000, 
                            'train': 74, 
                            'test': 69, 
                            'predict': 69}
    else:
        _FILE_PATTERN = ""
        _SPLITS_TO_SIZES = {'unlabeled': 0, 'train': 0, 'test': 0, 'predict': 0}

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
    print("************ {}".format(_FILE_PATTERN))
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
