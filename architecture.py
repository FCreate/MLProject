#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:34:47 2018

@author: mikolajbinkowski
"""
from functools import partial

import tensorflow as tf

from snops import batch_norm, conv2d, deconv2d, linear, lrelu, linear_one_hot
from misc import conv_sizes
# Generators
class Generator(object):
    def __init__(self, dim, c_dim, output_size, use_batch_norm, prefix='g_', scale=1.0, format='NCHW', is_train=True):
        self.used = False
        self.use_batch_norm = use_batch_norm
        self.dim = dim
        self.c_dim = c_dim
        self.output_size = output_size
        self.prefix = prefix
        self.scale = scale
        self.format = format
        self.is_train = is_train

        self.g_bn0 = self.make_bn(0)
        self.g_bn1 = self.make_bn(1)
        self.g_bn2 = self.make_bn(2)
        self.g_bn3 = self.make_bn(3)
        self.g_bn4 = self.make_bn(4)
        self.g_bn5 = self.make_bn(5)

    def make_bn(self, n):
        if self.use_batch_norm:
            bn = batch_norm(name='{}bn{}'.format(self.prefix, n),
                            format=self.format)
            return partial(bn, train=self.is_train)
        else:
            return lambda x: x

    def __call__(self, seed, batch_size, update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('generator') as scope:
            if self.used:
                scope.reuse_variables()
            self.used = True
            return self.network(seed, batch_size, update_collection)

    def network(self, seed, batch_size, update_collection):
        pass

    def data_format(self, batch_size, height, width, channel):
        if self.format == 'NCHW':
            return [batch_size, channel, height, width]
        elif self.format == 'NHWC':
            return [batch_size, height, width, channel]


class DCGANGenerator(Generator):
    def network(self, seed, batch_size, update_collection):
        s1, s2, s4, s8, s16 = conv_sizes(self.output_size, layers=4, stride=2)
        # 64, 32, 16, 8, 4 - for self.output_size = 64
        # default architecture
        # For Cramer: self.gf_dim = 64
        z_ = linear(seed, self.dim * 8 * s16 * s16, self.prefix + 'h0_lin', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False)   # project random noise seed and reshape

        h0 = tf.reshape(z_, self.data_format(batch_size, s16, s16, self.dim * 8))
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = deconv2d(h0, self.data_format(batch_size, s8, s8, self.dim*4), name=self.prefix + 'h1', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, self.data_format(batch_size, s4, s4, self.dim*2), name=self.prefix + 'h2', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = deconv2d(h2, self.data_format(batch_size, s2, s2, self.dim*1), name=self.prefix + 'h3', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = deconv2d(h3, self.data_format(batch_size, s1, s1, self.c_dim), name=self.prefix + 'h4', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format)
        return tf.nn.sigmoid(h4)


class SNGANGenerator(Generator):
    # DCGAN Generator used in 'Spectral Normalization in GANs', based on https://github.com/minhnhat93/tf-SNDCGAN/blob/master/net.py
    def network(self, seed, batch_size, update_collection):
        s1, s2, s4, s8, s16 = conv_sizes(self.output_size, layers=4, stride=2)
        z_ = linear(seed, self.dim * 8 * s8 * s8, self.prefix + 'h0_lin', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False)  # project random noise seed and reshape

        h0 = tf.reshape(z_, self.data_format(batch_size, s8, s8, self.dim * 8))
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = deconv2d(h0, self.data_format(batch_size, s4, s4, self.dim*4), name=self.prefix + 'h1', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, self.data_format(batch_size, s2, s2, self.dim*2), name=self.prefix + 'h2', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = deconv2d(h2, self.data_format(batch_size, s1, s1, self.dim*1), name=self.prefix + 'h3', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format)
        h3 = tf.nn.relu(self.g_bn3(h3))
        # SN dcgan generator implementation has smaller convolutional field and stride=1
        h4 = deconv2d(h3, self.data_format(batch_size, s1, s1, self.c_dim), k_h=3, k_w=3, d_h=1, d_w=1, name=self.prefix + 'h4', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format)
        return tf.nn.sigmoid(h4)


# Discriminator

class Discriminator(object):
    def __init__(self, dim, o_dim, use_batch_norm, prefix='d_', scale=1.0, format='NCHW', is_train=True):
        self.dim = dim
        self.o_dim = o_dim
        self.prefix = prefix
        self.used = False
        self.use_batch_norm = use_batch_norm
        self.scale = scale
        self.format = format
        self.is_train = is_train

        self.d_bn0 = self.make_bn(0)
        self.d_bn1 = self.make_bn(1)
        self.d_bn2 = self.make_bn(2)
        self.d_bn3 = self.make_bn(3)
        self.d_bn4 = self.make_bn(4)
        self.d_bn5 = self.make_bn(5)

    def make_bn(self, n, prefix=None):
        if prefix is None:
            prefix = self.prefix

        if self.use_batch_norm:
            bn = batch_norm(name='{}bn{}'.format(prefix, n),
                            format=self.format)
            return partial(bn, train=self.is_train)
        else:
            return lambda x: x

    def __call__(self, image, batch_size, return_layers=False,  update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope("discriminator") as scope:
            if self.used:
                scope.reuse_variables()
            self.used = True
            layers = self.network(image, batch_size, update_collection)
            if return_layers:
                return layers
            return layers['hF']

    def network(self, image, batch_size):
        pass

class DCGANDiscriminator(Discriminator):
    def network(self, image, batch_size, update_collection):
        o_dim = self.o_dim if (self.o_dim > 0) else 8 * self.dim
        h0 = lrelu(conv2d(image, self.dim, name=self.prefix + 'h0_conv', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format,with_singular_values=True))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.dim * 2, name=self.prefix + 'h1_conv', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format,with_singular_values=True)))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.dim * 4, name=self.prefix + 'h2_conv', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format,with_singular_values=True)))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.dim * 8, name=self.prefix + 'h3_conv', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False, data_format=self.format,with_singular_values=True)))
        hF = linear(tf.reshape(h3, [batch_size, -1]), o_dim, self.prefix + 'h4_lin', update_collection=update_collection, with_sn=False, scale=self.scale, with_learnable_sn_scale=False)
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'hF': hF}


class SNGANDiscriminator(Discriminator):
    # Discriminator used in 'Spectral Normalization in GANs', based on https://github.com/minhnhat93/tf-SNDCGAN/blob/master/net.py
    def network(self, image, batch_size, update_collection):
        c0_0 = lrelu(conv2d(image, 64, 3, 3, 1, 1, with_sn=False, with_learnable_sn_scale=False, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c0_0', data_format=self.format,with_singular_values=True))
        c0_1 = lrelu(conv2d(c0_0, 128, 4, 4, 2, 2, with_sn=False, with_learnable_sn_scale=False, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c0_1', data_format=self.format,with_singular_values=True))
        c1_0 = lrelu(conv2d(c0_1, 128, 3, 3, 1, 1, with_sn=False, with_learnable_sn_scale=False, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c1_0', data_format=self.format,with_singular_values=True))
        c1_1 = lrelu(conv2d(c1_0, 256, 4, 4, 2, 2, with_sn=False, with_learnable_sn_scale=False, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c1_1', data_format=self.format,with_singular_values=True))
        c2_0 = lrelu(conv2d(c1_1, 256, 3, 3, 1, 1, with_sn=False, with_learnable_sn_scale=False, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c2_0', data_format=self.format,with_singular_values=True))
        c2_1 = lrelu(conv2d(c2_0, 512, 4, 4, 2, 2, with_sn=False, with_learnable_sn_scale=False, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c2_1', data_format=self.format,with_singular_values=True))
        c3_0 = lrelu(conv2d(c2_1, 512, 3, 3, 1, 1, with_sn=False, with_learnable_sn_scale=False, update_collection=update_collection, stddev=0.02, name=self.prefix + 'c3_0', data_format=self.format,with_singular_values=True))
        c3_0 = tf.reshape(c3_0, [batch_size, -1])
        l4 = linear(c3_0, self.o_dim, with_sn=True, update_collection=update_collection, stddev=0.02, name=self.prefix + 'l4')
        return {'h0': c0_0, 'h1': c0_1, 'h2': c1_0, 'h3': c1_1, 'h4': c2_0, 'h5': c2_1, 'h6': c3_0, 'hF': l4}

def get_networks(architecture):
    print('architec', architecture)
    if architecture == 'dcgan':
        return DCGANGenerator, DCGANDiscriminator
    elif architecture == 'sngan':
        return SNGANGenerator, SNGANDiscriminator
    raise ValueError('Wrong architecture: "%s"' % architecture)
