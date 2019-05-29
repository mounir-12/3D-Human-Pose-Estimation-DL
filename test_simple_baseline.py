from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import copy

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import linear_model
import utils
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
tf.app.flags.DEFINE_boolean("predict_14", False, "predict 14 joints")
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")
tf.app.flags.DEFINE_boolean("max_norm", True, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_string("train_dir", "./experiments", "Training directory.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
# tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")
tf.app.flags.DEFINE_string("test_fname",   "./p2d_test.csv", "Test data file name")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

train_dir = FLAGS.train_dir

# checkpoint_path = os.path.join(train_dir, "checkpoint")

# if os.path.isfile(checkpoint_path):
#   dictionary = {}
#   with open(checkpoint_path) as f:
#     for line in f:
#       key, value = line.strip().split(': ')
#       dictionary[key] = value[1:-1]

#   latest_ckpt = dictionary['model_checkpoint_path']
# else:
#   raise ValueError("{0} does not seem to exist".format( checkpoint_path ) )

summaries_dir = os.path.join( train_dir, "log" )

os.system('mkdir -p {}'.format(summaries_dir))

def create_model( session, batch_size ):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """

  model = linear_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      FLAGS.predict_14,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")

  if ckpt and ckpt.model_checkpoint_path:
    model.saver.restore( session, ckpt.model_checkpoint_path)
  else:
    print("Could not find checkpoint. Aborting.")
    raise ValueError("Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model


def test():

  fname_test_out = 'simple_baseline_submission.csv'
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto(
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    model = create_model( sess, FLAGS.batch_size )

    encoder_inputs = model.get_test_data( FLAGS.test_fname)
    encoder_inputs = model.input_normalizer.transform(encoder_inputs)

    decoder_outputs = model.test_step(encoder_inputs, sess)

    decoder_outputs = model.output_normalizer.inverse_transform(decoder_outputs)

    utils.generate_submission_3d(decoder_outputs, fname_test_out)


def main(_):
    test()

if __name__ == "__main__":
  tf.app.run()