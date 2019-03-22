#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:11:46 2018

@author: mikolajbinkowski
"""
from __future__ import absolute_import, division, print_function
import os
import time
import lmdb
import io
import numpy as np
import tensorflow as tf
import misc
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

from tensorflow.python.ops.data_flow_ops import RecordInput, StagingArea
#from __future__ import absolute_import, division, print_function
import pathlib
import random



class NewJPEG(object):
    def __init__(self, output_size, c_dim, batch_size, data_dir, format='NCHW', with_labels=False, **kwargs):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.data_dir = data_dir
        self.batch_size = batch_size
        files = glob(os.path.join(self.data_dir, '*.jpg'))
        files = [str(path) for path in files]
        random.shuffle(files)
        path_ds = tf.data.Dataset.from_tensor_slices(files)
        image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        image_count = len(files)
        ds = image_ds.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds






class Pipeline(object):
    def __init__(self, output_size, c_dim, batch_size, data_dir, format='NCHW', with_labels=False, **kwargs):
        self.output_size = output_size
        self.c_dim = c_dim
        self.batch_size = batch_size
        self.read_batch = max(4000, batch_size * 10)
        self.read_count = 0
        self.data_dir = data_dir
        self.shape = [self.read_batch, self.output_size, self.output_size, self.c_dim]
        self.coord = None
        self.threads = None
        self.format = format
        if self.format == 'NCHW':
            self.shape = [self.read_batch,  self.c_dim, self.output_size, self.output_size]

    def _transform(self, x):
        return x

    def connect(self):
        assert hasattr(self, 'single_sample'), 'Pipeline needs to have single_sample defined before connecting'
        with tf.device('/cpu:0'):
            self.single_sample.set_shape(self.shape)
            ims = tf.train.shuffle_batch(
                [self.single_sample],
                self.batch_size,
                capacity=self.read_batch,
                min_after_dequeue=self.read_batch//8,
                num_threads=16,
                enqueue_many=len(self.shape) == 4)
            ims = self._transform(ims)
            images_shape = ims.get_shape()
            image_producer_stage = StagingArea(dtypes=[tf.float32], shapes=[images_shape])
            image_producer_op = image_producer_stage.put([ims])
            image_producer_stage_get = image_producer_stage.get()[0]
            images = tf.tuple([image_producer_stage_get], control_inputs=[image_producer_op])[0]
        return images

    def start(self, sess):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def stop(self):
        self.coord.request_stop()
        self.coord.join(self.threads)


class ConstantPipe(Pipeline):
    def __init__(self, *args, **kwargs):
        super(ConstantPipe, self).__init__(*args, **kwargs)
        stddev = 0.2
        if self.format == 'NCHW':
            out_shape = [self.batch_size, self.c_dim, self.output_size, self.output_size]
        elif self.format == 'NHWC':
            out_shape = [self.batch_size, self.output_size, self.output_size, self.c_dim]

        X = tf.get_variable('X', out_shape,
                            initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=False)
        self.images = X

    def connect(self):
        return self.images



def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  #image = tf.image.resize(image, [192, 192])
  #image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
    print (path)
    image = tf.read_file(path)
    return preprocess_image(image)
class JPEG(Pipeline):
    def __init__(self, *args,  **kwargs):
        super(JPEG, self).__init__(*args, **kwargs)
        self.data_dir = "/content/drive/My\ Drive/NewMLProject/Scaled-MMD-GAN/gan/data/JPEG/"
        #AUTOTUNE = tf.data.experimental.AUTOTUNE
        base_size = 160
        random_crop = 9
        files = glob(os.path.join(self.data_dir, '*.jpg'))
        print(files)
        files = [str(file) for file in files]
        print(self.data_dir)
        print(files)
        #files = tf.convert_to_tensor(files, dtype=tf.string)
        #tf.cast(files, dtype=tf.string)
        #  print(files)
        #filename_queue = tf.data.Dataset.from_tensor_slices(files).shuffle(tf.shape(files, out_type=tf.int64)[0])
        #filename_queue = tf.train.string_input_producer(files, shuffle=True)
        #image_ds = filename_queue.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        #reader = tf.WholeFileReader()
        #_, raw = reader.read(filename_queue)
        #decoded = tf.image.decode_jpeg(raw, channels=self.c_dim)  # HWC

        filename_queue = tf.train.string_input_producer(files, shuffle=True)
        reader = tf.WholeFileReader()
        _, raw = reader.read(filename_queue)
        image_ds = tf.image.decode_jpeg(raw, channels=self.c_dim)  # HWC

        bs = base_size + 2 * random_crop
        cropped = tf.image.resize_image_with_crop_or_pad(image_ds, bs, bs)
        if random_crop > 0:
            cropped = tf.image.random_flip_left_right(cropped)
            cropped = tf.random_crop(cropped, [base_size, base_size, self.c_dim])
        self.single_sample = cropped
        self.shape = [base_size, base_size, self.c_dim]

    def _transform(self, x):
        x = tf.image.resize_bilinear(x, (self.output_size, self.output_size))
        if self.format == 'NCHW':
            x = tf.transpose(x, [0, 3, 1, 2])
        return tf.cast(x, tf.float32)/255.


class Mnist(Pipeline):
    def __init__(self, *args, **kwargs):
        super(Mnist, self).__init__(*args, **kwargs)
        fd = open(os.path.join(self.data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if self.format == 'NCHW':
            trX = loaded[16:].reshape((60000, 1, 28, 28)).astype(np.float)
        elif self.format == 'NHWC':
            trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(self.data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if self.format == 'NCHW':
            teX = loaded[16:].reshape((10000, 1, 28, 28)).astype(np.float)
        elif self.format == 'NHWC':
            teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0).astype(np.float32) / 255.

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)

        queue = tf.train.input_producer(tf.constant(X), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)


class Cifar10(Pipeline):
    def __init__(self, *args, **kwargs):
        super(Cifar10, self).__init__(*args, **kwargs)
        self.categories = np.arange(10)

        batchesX, batchesY = [], []
        for batch in range(1, 6):
            pth = os.path.join(self.data_dir, 'data_batch_{}'.format(batch))
            labels, pixels = self.load_batch(pth)
            batchesX.append(pixels)
            batchesY.append(labels)
        trX = np.concatenate(batchesX, axis=0)

        _, teX = self.load_batch(os.path.join(self.data_dir, 'test_batch'))

        X = np.concatenate((trX, teX), axis=0).astype(np.float32) / 255.

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)

        queue = tf.train.input_producer(tf.constant(X), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)

    def load_batch(self, pth):
        if os.path.exists(pth):
            loaded = misc.unpickle(pth)
            labels = np.asarray(loaded['labels'])
            pixels = np.asarray(loaded['data'])
        elif os.path.exists(pth + '.bin'):
            loaded = np.fromfile(pth + '.bin', dtype=np.uint8).reshape(-1, 3073)
            labels = loaded[:, 0]
            pixels = loaded[:, 1:]
        else:
            raise ValueError("couldn't find {}".format(pth))

        idx = np.in1d(labels, self.categories)
        labels = labels[idx]
        pixels = pixels[idx].reshape(-1, 3, 32, 32)
        if self.format == 'NHWC':
            pixels = pixels.transpose(0, 2, 3, 1)
        return labels, pixels



def myhist(X, ax=plt, bins='auto', **kwargs):
    hist, bin_edges = np.histogram(X, bins=bins)
    hist = hist / hist.max()
    return ax.plot(
        np.c_[bin_edges, bin_edges].ravel(),
        np.r_[0, np.c_[hist, hist].ravel(), 0],
        **kwargs
    )


def get_pipeline(dataset, info):
    if dataset == 'mnist':
        return Mnist
    if dataset == 'cifar10':
        return Cifar10
    if dataset == 'JPEG':
        return JPEG
    else:
        raise Exception('invalid dataset: %s' % dataset)
