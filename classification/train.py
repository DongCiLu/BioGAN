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

import os
import math
from PIL import Image
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
layers = tf.contrib.layers
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops

import data_provider

flags = tf.flags


flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/mnist/',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_string('mode', 'train', 'Possible mode: [train predict visualize].')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('max_eval_steps', 20,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('num_predictions', 11250,
                     'number of images to predict labels.')

flags.DEFINE_string('prediction_dir', 'predicted_rosettes',
                     'directories to save predicted images.')

FLAGS = flags.FLAGS

def input_fn(split):
  images, labels, _ = data_provider.provide_data(
              split, FLAGS.batch_size, FLAGS.dataset_dir, 
              num_threads=4, mode="classification")
  return (images, labels)

_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)

def cnn_model(features, labels, mode):
  n_classes = 2
  trainable = True
  learning_rate = 1e-5
  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.fully_connected],
      activation_fn=_leaky_relu, normalizer_fn=layers.batch_norm):
      # activation_fn=tf.nn.leaky_relu, normalizer_fn=None):
    base_size = 1024
    conv1 = layers.conv2d(features, int(base_size / 32), 
            [4, 4], stride=2, trainable=trainable)
    conv2 = layers.conv2d(conv1, int(base_size / 16), 
            [4, 4], stride=2, trainable=trainable)
    conv3 = layers.conv2d(conv2, int(base_size / 8), 
            [4, 4], stride=2, trainable=trainable)
    conv4 = layers.conv2d(conv3, int(base_size / 4), 
            [4, 4], stride=2, trainable=trainable)
    conv5 = layers.conv2d(conv4, int(base_size / 2), 
            [4, 4], stride=2, trainable=trainable)
    flat = layers.flatten(conv5)
    fc1 = layers.fully_connected(flat, base_size)
    fc1_dropout = layers.dropout(fc1, 
            is_training=(mode==tf.estimator.ModeKeys.TRAIN))
    logits = layers.fully_connected(
            fc1_dropout, n_classes, activation_fn=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={
              'conv1': conv1,
              'class': predicted_classes,
              'prob': tf.nn.softmax(logits),
              'features': data_provider.float_image_to_uint8(features)
          })

    groundtruth_classes = tf.argmax(labels, 1)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss, 
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
      'accuracy':tf.metrics.accuracy(
          labels=groundtruth_classes, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def prime_powers(n):
  factors = set()
  for x in xrange(1, int(math.sqrt(n)) + 1):
    if n % x == 0:
      factors.add(int(x))
      factors.add(int(n // x))
  return sorted(factors)

def get_grid_dim(x):
  """
  Transforms x into product of two integers
  """
  factors = prime_powers(x)
  if len(factors) % 2 == 0:
    i = int(len(factors) / 2)
    return factors[i], factors[i - 1]

  i = len(factors) // 2
  return factors[i], factors[i]

def visualization_stack(imgs, mode):
  num_slices = imgs.shape[-1]
  vmin = np.min(imgs)
  vmax = np.max(imgs)
  grid_r, grid_c = get_grid_dim(num_slices)
  fig, axes = plt.subplots(min([grid_r, grid_c]), max([grid_r, grid_c]))
  for l, ax in enumerate(axes.flat):
      if mode == "weights":
          img = imgs[:, :, 0, l] 
          ax.imshow(img, vmin=vmin, vmax=vmax, 
                  interpolation='nearest', cmap='gray')
      elif mode == "activations":
          img = imgs[:, :, l] 
          ax.imshow(img, vmin=vmin, vmax=vmax, 
                  interpolation='bicubic', cmap='gray')
      else:
          raise ValueError('Image stack mode error!')
      ax.set_xticks([])
      ax.set_yticks([])
  plt.savefig("{}.png".format(mode), bbox_inches='tight')

def main(_):
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  # ws_file = "celegans-model/classification_nobn"
  ws_file = "celegans-model/unconditional3"
  ws = tf.estimator.WarmStartSettings(
          ckpt_to_initialize_from=ws_file, 
          # vars_to_warm_start=".*weights.*")
          # vars_to_warm_start="^(?!.*(BatchNorm))")
          vars_to_warm_start =
                  # "(Conv.*|fully_connected)/weights.*",
                  "((Conv.*|fully_connected)/weights.*)|Conv.*/BatchNorm.*",
          var_name_to_prev_var_name={
              "Conv/weights": "Discriminator/Conv/weights",
              "Conv_1/weights": "Discriminator/Conv_1/weights",
              "Conv_2/weights": "Discriminator/Conv_2/weights",
              "Conv_3/weights": "Discriminator/Conv_3/weights",
              "Conv_4/weights": "Discriminator/Conv_4/weights",
              "Conv/BatchNorm/beta": "Discriminator/Conv/BatchNorm/beta",
              "Conv_1/BatchNorm/beta": "Discriminator/Conv_1/BatchNorm/beta",
              "Conv_2/BatchNorm/beta": "Discriminator/Conv_2/BatchNorm/beta",
              "Conv_3/BatchNorm/beta": "Discriminator/Conv_3/BatchNorm/beta",
              "Conv_4/BatchNorm/beta": "Discriminator/Conv_4/BatchNorm/beta",
              "fully_connected/weights": 
                    "Discriminator/fully_connected/weights"})

  classifier = tf.estimator.Estimator(
          model_fn=cnn_model, 
          model_dir=FLAGS.train_log_dir,
          # warm_start_from=None)
          warm_start_from=ws)
  # debug_hook = tf_debug.TensorBoardDebugHook("dgx-dl03:7006")

  if FLAGS.mode == 'train':
      train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn('train'),
              max_steps=FLAGS.max_number_of_steps)
      eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn('test'),
              throttle_secs=5, start_delay_secs=5)

      tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

  elif FLAGS.mode == 'predict':
      if not tf.gfile.Exists(FLAGS.prediction_dir):
        tf.gfile.MakeDirs(FLAGS.prediction_dir)
        tf.gfile.MakeDirs("{}/{}".format(
            FLAGS.prediction_dir, "rosettes"))
        tf.gfile.MakeDirs("{}/{}".format(
            FLAGS.prediction_dir, "non-rosettes"))

      predictions = classifier.predict(input_fn=lambda:input_fn('predict'))
      for pred, cnt in zip(predictions, range(FLAGS.num_predictions)):
          rosette_image = np.array(pred['features'])[:,:,0]
          rosette_image = Image.fromarray(rosette_image)
          if pred['class'] == 1:
              rosette_image.save('{}/rosettes/test_{}.jpg'.format(
                  FLAGS.prediction_dir, cnt))
          else:
              rosette_image.save('{}/non-rosettes/test_{}.jpg'.format(
                  FLAGS.prediction_dir, cnt))

  elif FLAGS.mode == 'visualize':
      weights = classifier.get_variable_value("Conv/weights")
      visualization_stack(weights, "weights")
      predictions = classifier.predict(input_fn=lambda:input_fn('predict'))
      for pred, cnt in zip(predictions, range(FLAGS.num_predictions)):
          conv1 = pred['conv1']
          print ("*****************{}".format(pred['prob']))
          visualization_stack(conv1, "activations")
          # only print activation for one instance
          break
  
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
