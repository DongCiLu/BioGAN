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
"""Trains a classfier on Celegans data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import os
import math
from PIL import Image
from PIL import ImageDraw
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

flags.DEFINE_string('hyper_mode', 'regular', 
                    'Possible mode: [regular tiny].')

flags.DEFINE_string('mode', 'train', 
                    'Possible mode: [train predict visualize].')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('max_eval_steps', 20,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('num_predictions', 11250,
                     'number of images to predict labels.')

flags.DEFINE_integer('warm_start', 0,
                     'cold start or warm start.')

flags.DEFINE_string('prediction_dir', 'predicted_rosettes',
                     'directories to save predicted images.')

flags.DEFINE_string('visualization_dir', 'visualization',
                     'directories to save visualization images.')

FLAGS = flags.FLAGS

Encoding_para = 10000 # for encoding visualization coordinates

def input_fn(split, mode):
    print("hyper mode: {}".format(mode))
    images, labels, filenames, _ = data_provider.provide_data(
                split, FLAGS.batch_size, FLAGS.dataset_dir, 
                num_threads=1, mode=mode)
    features = {'images': images, 'filenames': filenames}
    return (features, labels)
  
def input_fn_visualization_occ(base_name):
    image_size = 128
    occ_size = 32
    stride_size = 4
    n_channels = 1
    occlusion_val = 0
    n_images = (int((image_size - occ_size) / stride_size)) ** 2;
    # n_images = 1;
    print("number of images for visualization: {}".format(n_images))
    images = np.empty(shape=(n_images, image_size, image_size, n_channels),
            dtype=np.float32)
    # filenames = np.empty(shape=(n_images), dtype='str')
    filenames = np.empty(shape=(n_images), dtype=np.int32)
    image_occlusion = Image.new('L', (occ_size, occ_size))
    for x in range(occ_size):
        for y in range(occ_size):
            image_occlusion.putpixel((x, y), occlusion_val)
    base_image = Image.open(base_name)
    image_cnt = 0
    for x in range(0, image_size - occ_size, stride_size):
        for y in range(0, image_size - occ_size, stride_size):
            occluded_image = base_image.copy()
            occluded_image.paste(image_occlusion, (x, y))
            # occluded_image.save('occlusion/occ_{}_{}.jpg'.format(x, y))
            image_array = np.array(occluded_image)
            image_array = (image_array - 128.0) / 128.0
            image_array = np.float32(image_array)
            image_array = np.expand_dims(image_array, axis=2)
            images[image_cnt] = image_array
            # magic encoding, using filename to save x and y coordinates
            filenames[image_cnt] = x * Encoding_para + y 
            image_cnt += 1
    features = {'images': images, 'filenames': filenames}
    image_queue = tf.estimator.inputs.numpy_input_fn(
            features, batch_size=FLAGS.batch_size, shuffle=False, num_epochs=1)
    return image_queue
    
def input_fn_visualization_patch(base_name):
    image_size = 128
    patch_size = 32
    stride_size = 4
    n_channels = 1
    n_images = (int((image_size - patch_size) / stride_size)) ** 2;
    print("number of images for visualization: {}".format(n_images))
    images = np.empty(shape=(n_images, patch_size, patch_size, n_channels),
            dtype=np.float32)
    # filenames = np.empty(shape=(n_images), dtype='str')
    filenames = np.empty(shape=(n_images), dtype=np.int32)
    base_image = Image.open(base_name)
    image_cnt = 0
    for x in range(0, image_size - patch_size, stride_size):
        for y in range(0, image_size - patch_size, stride_size):
            patch = base_image.crop((x, y, x + patch_size, y + patch_size))
            image_array = np.array(patch)
            image_array = (image_array - 128.0) / 128.0
            image_array = np.float32(image_array)
            image_array = np.expand_dims(image_array, axis=2)
            images[image_cnt] = image_array
            # filenames[image_cnt] = 'patch_{}_{}.jpg'.format(x, y)
            # magic encoding, using filename to save x and y coordinates
            filenames[image_cnt] = x * Encoding_para + y 
            image_cnt += 1
    features = {'images': images, 'filenames': filenames}
    image_queue = tf.estimator.inputs.numpy_input_fn(
            features, batch_size=FLAGS.batch_size, shuffle=False, num_epochs=1)
    return image_queue
  
  
_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)
  
def cnn_model(features, labels, mode):
    n_classes = 2
    trainable = True
    learning_rate = 1e-5
    images = features['images']
    filenames = features['filenames']
    # setup batch normalization
    if mode == tf.estimator.ModeKeys.TRAIN:
        norm_params={'is_training':True}
    else:
        norm_params={'is_training':False,
                     'updates_collections': None}
    # create the network
    with tf.contrib.framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=_leaky_relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=norm_params):
        # activation_fn=tf.nn.leaky_relu, normalizer_fn=None):
      if FLAGS.hyper_mode == "regular":
          print("set up the network in regular mode")
          base_size = 1024
          conv1 = layers.conv2d(images, int(base_size / 32), 
                  [4, 4], stride=2, trainable=trainable)
          conv2 = layers.conv2d(conv1, int(base_size / 16), 
                  [4, 4], stride=2, trainable=trainable)
      elif FLAGS.hyper_mode == "tiny":
          print("set up the network in tiny mode")
          # bypass first two layer.
          base_size = int(1024 / 4)
          # base_size = 1024
          conv2 = images
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
  
      if FLAGS.hyper_mode == "regular":
          visual_sample = conv1
      elif FLAGS.hyper_mode == "tiny":
          visual_sample = conv3
  
      predicted_classes = tf.argmax(logits, 1)
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'visual_sample': visual_sample,
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits),
                'images': data_provider.float_image_to_uint8(images),
                'filenames': filenames
            })
  
      groundtruth_classes = tf.argmax(labels, 1)
      loss = tf.losses.softmax_cross_entropy(
              onehot_labels=labels, logits=logits)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, 
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)
  
      precision = tf.metrics.precision(
            labels=groundtruth_classes, predictions=predicted_classes)
      recall = tf.metrics.recall(
            labels=groundtruth_classes, predictions=predicted_classes)
  
      eval_metric_ops = {
        'eval/accuracy': tf.metrics.accuracy(
            labels=groundtruth_classes, predictions=predicted_classes),
        'eval/precision': precision,
        'eval/recall': recall
        # 'f1_score': f1_score
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
  
def get_prefix(name):
      dir_index = name.rfind('/')
      extension_index = name.rfind('.')
      prefix = name[dir_index + 1 : extension_index]
      return prefix
  
def save_visual_stack(imgs, mode):
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
    plt.savefig("{}/{}.png".format(
        FLAGS.visualization_dir, mode), bbox_inches='tight')
  
def save_visual_image(base_name, predictions, thresholds):
    base_image = Image.open(base_name)
    rgb_base_images = []
    draws = []
    for threshold in thresholds:
        rgb_base_images.append(Image.new("RGB", base_image.size))
        rgb_base_images[-1].paste(base_image)
        draws.append(ImageDraw.Draw(rgb_base_images[-1]))
    has_rosette = False
    for pred in predictions:
        encode = pred['filenames']
        prob = pred['prob']
        pred_class = pred['class']
        x = encode / Encoding_para
        y = encode % Encoding_para
        for index, threshold in enumerate(thresholds):
            print("{},{}: {}".format(int(x), int(y), prob[0]))
            if (prob[0] > threshold) :
                has_rosette = True
                draws[index].rectangle(
                        [x, y, x + 32, y + 32], outline=(255,0,0))
    # discover mode (1 threshold) or test mode (multiple threshold)
    if has_rosette or len(thresholds) > 1: 
        for rgb_base_image, threshold in zip(rgb_base_images, thresholds):
            rgb_base_image.save('{}/{}_{}.jpg'.format(
                FLAGS.visualization_dir, get_prefix(base_name), threshold))
  
def main(_):
    if not tf.gfile.Exists(FLAGS.train_log_dir):
      tf.gfile.MakeDirs(FLAGS.train_log_dir)
  
    if FLAGS.hyper_mode == 'regular':
        # ws_file = "celegans-model/classification_nobn"
        ws_file = "celegans-model/unconditional3"
        var_names = {
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
                      "Discriminator/fully_connected/weights"}
    elif FLAGS.hyper_mode == 'tiny':
        ws_file = "celegans-model/tinygan-test"
        var_names = {
                "Conv/weights": "Discriminator/Conv/weights",
                "Conv_1/weights": "Discriminator/Conv_1/weights",
                "Conv_2/weights": "Discriminator/Conv_2/weights",
                "Conv/BatchNorm/beta": "Discriminator/Conv/BatchNorm/beta",
                "Conv_1/BatchNorm/beta": "Discriminator/Conv_1/BatchNorm/beta",
                "Conv_2/BatchNorm/beta": "Discriminator/Conv_2/BatchNorm/beta",
                "fully_connected/weights": 
                      "Discriminator/fully_connected/weights"}

    ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=ws_file, 
            # vars_to_warm_start=".*weights.*")
            # vars_to_warm_start="^(?!.*(BatchNorm))")
            vars_to_warm_start =
                    # "(Conv.*|fully_connected)/weights.*",
                    "((Conv.*|fully_connected)/weights.*)|Conv.*/BatchNorm.*",
            var_name_to_prev_var_name=var_names)

    if FLAGS.warm_start == 0:
        ws = None

    classifier = tf.estimator.Estimator(
            model_fn=cnn_model, 
            model_dir=FLAGS.train_log_dir,
            warm_start_from=ws)
    # debug_hook = tf_debug.TensorBoardDebugHook("dgx-dl03:7006")
  
    if FLAGS.mode == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=
                lambda: input_fn('train', FLAGS.hyper_mode),
                max_steps=FLAGS.max_number_of_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=
                lambda: input_fn('test', FLAGS.hyper_mode),
                throttle_secs=5, start_delay_secs=5)
  
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
  
    elif FLAGS.mode == 'predict':
        if not tf.gfile.Exists(FLAGS.prediction_dir):
          tf.gfile.MakeDirs(FLAGS.prediction_dir)
          tf.gfile.MakeDirs("{}/{}".format(
              FLAGS.prediction_dir, "rosettes"))
          tf.gfile.MakeDirs("{}/{}".format(
              FLAGS.prediction_dir, "non-rosettes"))
  
        ros_cnt = 0
        total_cnt = 0
        predictions = classifier.predict(input_fn=
                lambda:input_fn('predict', FLAGS.hyper_mode))
        for pred, cnt in zip(predictions, range(FLAGS.num_predictions)):
            rosette_image = np.array(pred['images'])[:,:,0]
            rosette_image = Image.fromarray(rosette_image)
            fn_prefix = get_prefix(pred['filenames'])
            total_cnt += 1
            if pred['class'] == 1:
                ros_cnt += 1
                rosette_image.save('{}/rosettes/{}_{}.jpg'.format(
                    FLAGS.prediction_dir, fn_prefix, pred['prob']))
            else:
                rosette_image.save('{}/non-rosettes/{}_{}.jpg'.format(
                    FLAGS.prediction_dir, fn_prefix, pred['prob']))
  
        print('number of rosettes in the dataset: {}/{}'
                .format(ros_cnt, total_cnt))
  
    elif FLAGS.mode == 'visualize':
        if not tf.gfile.Exists('{}'.format(FLAGS.visualization_dir)):
          tf.gfile.MakeDirs('{}'.format(FLAGS.visualization_dir))
        visual_input = 'visual_input'
  
        # visualize weight and activation
        weights = classifier.get_variable_value("Conv/weights")
        save_visual_stack(weights, "weights")
        predictions = classifier.predict(input_fn=
                lambda:input_fn('predict', FLAGS.hyper_mode))
        for pred, cnt in zip(predictions, range(FLAGS.num_predictions)):
            visual_sample = pred['visual_sample']
            save_visual_stack(visual_sample, "activations")
            # only print activation for one layer
            break
  
        # visualize occulation & patch
        thresholds = [0.9]
        # visualize the image one by one, throughput is low
        for subdir, dirs, files in os.walk(visual_input):
            for f in files:
                filename = os.path.join(subdir, f)
                # occulation still uses 128x128 network,
                # however, patch uses 32x32 network
                # predictions = classifier.predict(
                      # input_fn_visualization_occ(filename))
                predictions = classifier.predict(
                      input_fn_visualization_patch(filename))
                save_visual_image(filename, predictions, thresholds)
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
