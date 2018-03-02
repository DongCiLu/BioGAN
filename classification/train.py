# Copyright 2017 The ensorFlow Authors. All Rights Reserved.
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
"""Trains a generator on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf
layers = tf.contrib.layers
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops

import data_provider
import os

flags = tf.flags


flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/mnist/',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_string('mode', 'train', 'Possible mode: [train predict].')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('max_eval_steps', 20,
                     'The maximum number of gradient steps.')

FLAGS = flags.FLAGS

def input_fn(split):
  images, labels, _ = data_provider.provide_data(
              split, FLAGS.batch_size, FLAGS.dataset_dir, 
              num_threads=4, mode="classification")
  return (images, labels)

def cnn_model(features, labels, mode):
  n_classes = 2
  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.fully_connected],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm):
    conv1 = layers.conv2d(features, 8, [4, 4], stride=2)
    conv2 = layers.conv2d(conv1, 16, [4, 4], stride=2)
    conv3 = layers.conv2d(conv2, 32, [4, 4], stride=2)
    conv4 = layers.conv2d(conv3, 64, [4, 4], stride=2)
    flat = layers.flatten(conv4)
    fc1 = layers.fully_connected(flat, 256)
    fc1_dropout = layers.dropout(fc1, 
            is_training=(mode==tf.estimator.ModeKeys.TRAIN))
    logits = layers.fully_connected(
            fc1_dropout, n_classes, activation_fn=None)

    # tf.contrib.layers.summarize_activations(
            # conv1, summarizer=tf.contrib.layers.summarize_activation)
    # tf.contrib.layers.summarize_activation(conv1)
    print (os.path.split(conv1.name)[0] + "/" + ops.GraphKeys.WEIGHTS)
    conv1_op = tf.get_default_graph().get_operation_by_name("Conv/Conv2D")
    wconv1 = tf.get_default_graph().get_tensor_by_name(
            os.path.split(conv1.name)[0] + '/weights/read')
    print (wconv1)
    tf.contrib.layers.summarize_collection(ops.GraphKeys.WEIGHTS, conv1)

    predicted_classes = tf.argmax(logits, 1)
    groundtruth_classes = tf.argmax(labels, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={
              'class': predicted_classes,
              'prob': tf.nn.softmax(logits)
          })

    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(loss, 
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
      'accuracy':tf.metrics.accuracy(
          labels=groundtruth_classes, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  classifier = tf.estimator.Estimator(
          model_fn=cnn_model, model_dir=FLAGS.train_log_dir)
  # debug_hook = tf_debug.TensorBoardDebugHook("dgx-dl03:7006")

  train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn('train'),
          max_steps=FLAGS.max_number_of_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn('test'),
          throttle_secs=3, start_delay_secs=3)

  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
  
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
