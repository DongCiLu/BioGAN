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
"""Networks for MNIST example using TFGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from slim.nets import dcgan

ds = tf.contrib.distributions
layers = tf.contrib.layers
tfgan = tf.contrib.gan


def _generator_helper(
      noise, is_conditional, one_hot_labels, weight_decay):
    """Core MNIST generator.
  
    This function is reused between the different GAN modes (unconditional,
    conditional, etc).
  
    Args:
      noise: A 2D Tensor of shape [batch size, noise dim].
      is_conditional: Whether to condition on labels.
      one_hot_labels: Optional labels for conditioning.
      weight_decay: The value of the l2 weight decay.
  
    Returns:
      A generated image in the range [-1, 1].
    """
    with tf.contrib.framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        # activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
      base_size = 1024
      net = layers.fully_connected(noise, base_size)
      # net = layers.dropout(net)
      if is_conditional:
        net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
      net = layers.fully_connected(net, 8 * 8 * int(base_size / 2))
      net = tf.reshape(net, [-1, 8, 8, int(base_size / 2)])
      net = layers.conv2d_transpose(net, int(base_size / 4), [4, 4], stride=2)
      net = layers.conv2d_transpose(net, int(base_size / 8), [4, 4], stride=2)
      net = layers.conv2d_transpose(net, int(base_size / 16), [4, 4], stride=2)
      net = layers.conv2d_transpose(net, int(base_size / 32), [4, 4], stride=2)
      net = layers.conv2d(
          net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)
  
      return net
  
def multiple_generator(inputs, is_conditional=False, weight_decay=2.5e-5):
    if is_conditional:
      noise, one_hot_labels = inputs
    else:
      noise = inputs
    with tf.contrib.framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
      half_slice = int(int(noise.shape[1]) / 2)
      batch_size = int(noise.shape[0])
      noise1 = tf.slice(noise, [0,0], [batch_size, half_slice])
      noise2 = tf.slice(noise, [0,half_slice], [batch_size, half_slice])
      net1 = layers.fully_connected(noise1, 1024)
      net2 = layers.fully_connected(noise2, 1024)
      if is_conditional:
        net1 = tfgan.features.condition_tensor_from_onehot(
                net1, one_hot_labels)
        net2 = tfgan.features.condition_tensor_from_onehot(
                net2, one_hot_labels)
      net2 = tf.concat([net1,net2], 1)
      net1 = layers.fully_connected(net1, 8 * 8 * 512)
      net2 = layers.fully_connected(net2, 8 * 8 * 512)
      net1 = tf.reshape(net1, [-1, 8, 8, 512])
      net2 = tf.reshape(net2, [-1, 8, 8, 512])
      net2 = tf.concat([net1,net2], 3)
      net1 = layers.conv2d_transpose(net1, 256, [4, 4], stride=2)
      net2 = layers.conv2d_transpose(net2, 256, [4, 4], stride=2)
      net2 = tf.concat([net1,net2], 3)
      net1 = layers.conv2d_transpose(net1, 128, [4, 4], stride=2)
      net2 = layers.conv2d_transpose(net2, 128, [4, 4], stride=2)
      net2 = tf.concat([net1,net2], 3)
      net1 = layers.conv2d_transpose(net1, 64, [4, 4], stride=2)
      net2 = layers.conv2d_transpose(net2, 64, [4, 4], stride=2)
      net2 = tf.concat([net1,net2], 3)
      net1 = layers.conv2d_transpose(net1, 32, [4, 4], stride=2)
      net2 = layers.conv2d_transpose(net2, 32, [4, 4], stride=2)
      net2 = tf.concat([net1,net2], 3)
      # Make sure that generator output is in the same range as `inputs`
      # ie [-1, 1].
      net1 = layers.conv2d(
          net1, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)
      net2 = layers.conv2d(
          net2, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)
      net = tf.concat([net1,net2], 2)
      print ("shape of output from G: {}".format(net.shape))
      return net
  
  
def unconditional_generator(noise, weight_decay=2.5e-5):
    """Generator to produce unconditional MNIST images.
  
    Args:
      noise: A single Tensor representing noise.
      weight_decay: The value of the l2 weight decay.
  
    Returns:
      A generated image in the range [-1, 1].
    """
    # image, _ = dcgan.generator(noise, final_size=128, num_outputs=1)
  
    # return tf.tanh(image)
    return _generator_helper(noise, False, None, weight_decay)
  
  
def conditional_generator(inputs, weight_decay=2.5e-5):
    """Generator to produce MNIST images conditioned on class.
  
    Args:
      inputs: A 2-tuple of Tensors (noise, one_hot_labels).
      weight_decay: The value of the l2 weight decay.
  
    Returns:
      A generated image in the range [-1, 1].
    """
    noise, one_hot_labels = inputs
    return _generator_helper(noise, True, one_hot_labels, weight_decay)
  
  
def infogan_generator(inputs, categorical_dim, weight_decay=2.5e-5):
    """InfoGAN generator network on MNIST digits.
  
    Based on a paper https://arxiv.org/abs/1606.03657, their code
    https://github.com/openai/InfoGAN, and code by pooleb@.
  
    Args:
      inputs: A 3-tuple of Tensors (unstructured_noise, categorical structured
        noise, continuous structured noise). `inputs[0]` and `inputs[2]` must be
        2D, and `inputs[1]` must be 1D. All must have the same first dimension.
      categorical_dim: Dimensions of the incompressible categorical noise.
      weight_decay: The value of the l2 weight decay.
  
    Returns:
      A generated image in the range [-1, 1].
    """
    unstructured_noise, cat_noise, cont_noise = inputs
    cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
    all_noise = tf.concat(
        [unstructured_noise, cat_noise_onehot, cont_noise], axis=1)
    return _generator_helper(all_noise, False, None, weight_decay)
  
  
_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)
  
  
def _discriminator_helper(img, is_conditional, one_hot_labels, weight_decay):
    """Core MNIST discriminator.
  
    This function is reused between the different GAN modes (unconditional,
    conditional, etc).
  
    Args:
      img: Real or generated MNIST digits. Should be in the range [-1, 1].
      is_conditional: Whether to condition on labels.
      one_hot_labels: Labels to optionally condition the network on.
      weight_decay: The L2 weight decay.
  
    Returns:
      Final fully connected discriminator layer. [batch_size, 1024].
    """
    with tf.contrib.framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        # activation_fn=_leaky_relu, normalizer_fn=None,
        # activation_fn=tf.nn.leaky_relu, normalizer_fn=None,
        # activation_fn=_leaky_relu, normalizer_fn=layers.batch_norm,
        activation_fn=tf.nn.leaky_relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
      base_size = 1024
      net = layers.conv2d(img, int(base_size / 32), [4, 4], stride=2)
      net = layers.conv2d(net, int(base_size / 16), [4, 4], stride=2)
      net = layers.conv2d(net, int(base_size / 8), [4, 4], stride=2)
      net = layers.conv2d(net, int(base_size / 4), [4, 4], stride=2)
      net = layers.conv2d(net, int(base_size / 2), [4, 4], stride=2)
      net = layers.flatten(net)
      if is_conditional:
        net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
      net = layers.fully_connected(
              net, base_size, normalizer_fn=layers.layer_norm)
  
      return net
  
  
def multiple_discriminator(img, unused_conditioning, weight_decay=2.5e-5):
    """Discriminator network on unconditional MNIST digits.
  
    Args:
      img: Real or generated MNIST digits. Should be in the range [-1, 1].
      unused_conditioning: The TFGAN API can help with conditional GANs, which
        would require extra `condition` information to both the generator and the
        discriminator. Since this example is not conditional, we do not use this
        argument.
      weight_decay: The L2 weight decay.
  
    Returns:
      Logits for the probability that the image is real.
    """
    with tf.contrib.framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        # activation_fn=_leaky_relu, normalizer_fn=None,
        activation_fn=tf.nn.leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
      print("shape 1: {}".format(img.shape))
      net = layers.conv2d(img, 32, [4, 4], stride=2)
      print("shape 2: {}".format(net.shape))
      net = layers.conv2d(net, 64, [4, 4], stride=2)
      print("shape 3: {}".format(net.shape))
      net = layers.conv2d(net, 128, [4, 4], stride=2)
      print("shape 4: {}".format(net.shape))
      net = layers.conv2d(net, 256, [4, 4], stride=2)
      print("shape 5: {}".format(net.shape))
      net = layers.flatten(net)
      print("shape 6: {}".format(net.shape))
      net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
      print("shape 7: {}".format(net.shape))
  
    return layers.linear(net, 1)
  
def unconditional_discriminator(img, unused_conditioning, weight_decay=2.5e-5):
    """Discriminator network on unconditional MNIST digits.
  
    Args:
      img: Real or generated MNIST digits. Should be in the range [-1, 1].
      unused_conditioning: The TFGAN API can help with conditional GANs, which
        would require extra `condition` information to both the generator and the
        discriminator. Since this example is not conditional, we do not use this
        argument.
      weight_decay: The L2 weight decay.
  
    Returns:
      Logits for the probability that the image is real.
    """
    # logits, _ = dcgan.discriminator(img)
    # return logits
    net = _discriminator_helper(img, False, None, weight_decay)
    return layers.linear(net, 1)
  
  
def conditional_discriminator(img, conditioning, weight_decay=2.5e-5):
    """Conditional discriminator network on MNIST digits.
  
    Args:
      img: Real or generated MNIST digits. Should be in the range [-1, 1].
      conditioning: A 2-tuple of Tensors representing (noise, one_hot_labels).
      weight_decay: The L2 weight decay.
  
    Returns:
      Logits for the probability that the image is real.
    """
    _, one_hot_labels = conditioning
    net = _discriminator_helper(img, True, one_hot_labels, weight_decay)
    return layers.linear(net, 1)
  
def infogan_discriminator(img, unused_conditioning, weight_decay=2.5e-5,
                            categorical_dim=10, continuous_dim=2):
    """InfoGAN discriminator network on MNIST digits.
  
    Based on a paper https://arxiv.org/abs/1606.03657, their code
    https://github.com/openai/InfoGAN, and code by pooleb@.
  
    Args:
      img: Real or generated MNIST digits. Should be in the range [-1, 1].
      unused_conditioning: The TFGAN API can help with conditional GANs, which
        would require extra `condition` information to both the generator and the
        discriminator. Since this example is not conditional, we do not use this
        argument.
      weight_decay: The L2 weight decay.
      categorical_dim: Dimensions of the incompressible categorical noise.
      continuous_dim: Dimensions of the incompressible continuous noise.
  
    Returns:
      Logits for the probability that the image is real, and a list of posterior
      distributions for each of the noise vectors.
    """
    net = _discriminator_helper(img, False, None, weight_decay)
    logits_real = layers.fully_connected(net, 1, activation_fn=None)
  
    # Recognition network for latent variables has an additional layer
    with tf.contrib.framework.arg_scope([layers.batch_norm], is_training=False):
      encoder = layers.fully_connected(net, 128, normalizer_fn=layers.batch_norm,
                                       activation_fn=_leaky_relu)
  
    # Compute logits for each category of categorical latent.
    logits_cat = layers.fully_connected(
        encoder, categorical_dim, activation_fn=None)
    q_cat = ds.Categorical(logits_cat)
  
    # Compute mean for Gaussian posterior of continuous latents.
    mu_cont = layers.fully_connected(encoder, continuous_dim, activation_fn=None)
    sigma_cont = tf.ones_like(mu_cont)
    q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)
  
    return logits_real, [q_cat, q_cont]
