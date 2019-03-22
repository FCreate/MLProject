from __future__ import absolute_import, division, print_function
import tensorflow as tf
import argparse
import os
from config import Config

def model_class(name):
    name = name.lower()
    if name == 'mmd':
        from model import MMD_GAN as Model
    elif name == 'smmd':
        from smmd import SMMD as Model
    else:
        raise ValueError("unknown model {}".format(name))
    return Model

parser = argparse.ArgumentParser()


def make_flags(args=None):
    global config
    FLAGS = parser.parse_args(args)
    config.dataset = FLAGS.dataset
    config.architecture = FLAGS.architecture
    config.model = FLAGS.model
    if FLAGS.dataset == 'mnist':
        config.output_size = 28
        config.c_dim = 1
    elif FLAGS.dataset == 'cifar10':
        config.output_size = 32
        config.c_dim = 3
    elif FLAGS.dataset =='JPEG':
        config.c_dim = 3

    return config


def add_arg(name, **kwargs):
    "Convenience to handle reasonable names for args as well as crappy ones."
    assert name[0] == '-'
    assert '-' not in name[1:]
    nice_name = '--' + name[1:].replace('_', '-')
    return parser.add_argument(name, nice_name, **kwargs)

add_arg('-architecture',                default="dcgan",        type=str,       help='The name of the architecture [*dcgan*, g-resnet5, dcgan5]')
add_arg('-model',                       default="smmd",          type=str,       help='The model type [*mmd*, smmd, swgan, wgan_gp]')
add_arg('-dataset',                     default="mnist",      type=str,       help='The name of the dataset [celebA, mnist, lsun, *cifar10*, imagenet]')

def main(_):
    tf.enable_eager_execution()
    global FLAGS
    global config
    #pp.pprint(vars(FLAGS))

    sess_config = tf.ConfigProto(
        device_count={"CPU": 3},
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    Model = model_class(config.model)
    with tf.Session(config=sess_config) as sess:
        gan = Model(sess, config=config)
        gan.train()
        gan.sess.close()


if __name__ == '__main__':
    config = Config()
    config = make_flags()
    tf.app.run()
