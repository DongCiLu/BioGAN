# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image

import tensorflow as tf

import celegans
import data_provider

class DataProviderTest(tf.test.TestCase):

  def test_celegans_data_reading(self):
    dataset_dir = "./celegans-128-data"
    # dataset_dir = "./celegans-mnist-data"
    # celegans.config_dataset("128_1to1old")
    batch_size = 10
    images, labels, filenames, num_samples = \
            data_provider.provide_data(
            'train', batch_size, dataset_dir, 
            mode="classification", data_config="128_1.0")

    with self.test_session() as sess:
      with tf.contrib.slim.queues.QueueRunners(sess):
        images, labels, filenames = \
                sess.run([images, labels, filenames])
        '''
            image = np.array(image[:,:,0])
            image = image * 128 + 128
            image = Image.fromarray(image.astype(np.uint8))
            image.save("test{}.jpg".format(cnt))
        '''


if __name__ == '__main__':
  tf.test.main()
