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

import data_provider

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
    net = layers.conv2d(features, 8, [4, 4], stride=2)
    net = layers.conv2d(net, 16, [4, 4], stride=2)
    net = layers.conv2d(net, 32, [4, 4], stride=2)
    net = layers.conv2d(net, 64, [4, 4], stride=2)
    net = layers.flatten(net)
    net = layers.fully_connected(net, 256)
    logits = layers.fully_connected(
            net, n_classes, activation_fn=None)

    # print ('number of trainable variables: {}'.format(
        # np.sum([np.prod(v.get_shape().as_list()) 
        # for v in tf.trainable_variables()])))

    predicted_classes = tf.argmax(logits, 1)
    n_pred_false = tf.reduce_sum(predicted_classes)
    groundtruth_classes = tf.argmax(labels, 1)
    n_gt_false = tf.reduce_sum(predicted_classes)
    tf.summary.scalar("pred_false", n_pred_false)
    tf.summary.scalar("gt_false", n_gt_false)
    tf.Print(labels, [labels], message="this is labels: ")
    tf.Print(logits, [logits], message="this is logits: ")
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={
              'class': predicted_classes,
              'prob': tf.nn.softmax(logits)
          })

    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits)
    tf.Print(loss, [loss], message="this is loss: ")
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

  '''
  if FLAGS.mode == 'train':
      for i in range(10):
          classifier.train(lambda: input_fn('train'), 
                  steps=FLAGS.max_number_of_steps / 10)
                  # hooks=[debug_hook])
          classifier.evaluate(lambda: input_fn('test'),
                  steps=FLAGS.max_eval_steps)
                  # hooks=[debug_hook])
  '''
  train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn('train'),
          max_steps=FLAGS.max_number_of_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn('test'),
          throttle_secs=3, start_delay_secs=3)

  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
  
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
