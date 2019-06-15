import math
import os, glob
import random
import sys
import time
import h5py

import numpy as np
import tensorflow as tf

import linear_model
import utils

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")
tf.app.flags.DEFINE_boolean("max_norm", True, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability. 1 means no dropout")

# Directories
list_of_files = glob.glob(os.path.join(".", "log_SB", "*"))
tf.app.flags.DEFINE_string("train_dir", max(list_of_files, key=os.path.getctime), "Training directory.") # latest created dir for latest experiment

FLAGS = tf.app.flags.FLAGS

train_dir = FLAGS.train_dir
print("\n")
print(train_dir)
print("\n")

summaries_dir = os.path.join( train_dir, "checkpoints" )

def create_model( session, batch_size ):
  """
  Loads model in a session

  Args
    session: tensorflow session
    batch_size: integer. Number of examples in each batch
  Returns
    model: The loaded model
  """

  model = linear_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      dtype=tf.float32)

  to_restore = tf.train.latest_checkpoint(summaries_dir)
  print("\nRestoring model from {}\n".format(to_restore))
  model.saver.restore(session, to_restore) # restore model from last checkpoint

  return model


def test():

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = "0"
  with tf.Session(config=config) as sess:

    model = create_model( sess, FLAGS.batch_size )

    list_of_experiments = glob.glob(os.path.join(".", "log_HG2D", "*"))
    p2d_dir = os.path.join(max(list_of_experiments, key=os.path.getctime), "predictions") # latest created dir for latest experiment
    list_of_preds = glob.glob(os.path.join(p2d_dir, "*"))
    
    print("\n")
    for test_fname in list_of_preds:
        test_fname = test_fname.split("/")[-1]
        print("Converting {} to p3d".format(test_fname))
        a = time.time()
        
        encoder_inputs = model.get_test_data(os.path.join(p2d_dir, test_fname))
        encoder_inputs = model.input_normalizer.transform(encoder_inputs)

        decoder_outputs = model.test_step(encoder_inputs, sess)

        decoder_outputs = model.output_normalizer.inverse_transform(decoder_outputs)

        for i in range(decoder_outputs.shape[0]):
          temp = decoder_outputs[i].reshape(17,3)
          temp -= temp[0]
          decoder_outputs[i] = temp.reshape(17*3)

        print("Done in {} s".format(time.time() - a))

        experiment = p2d_dir.split("/")[-2]
        submissions_dir = os.path.join(train_dir, "submissions_for_{}".format(experiment))
        if not os.path.exists(submissions_dir):
            os.makedirs(submissions_dir)
        
        step_nb = test_fname.split(".")[-2].split("_")[-1]
        out_file = os.path.join(submissions_dir, "p3d_step_{}.csv.gz".format(step_nb))
        print("saving at: {}".format(out_file))
        print("\n\n")
        utils.generate_submission_3d(decoder_outputs, out_file)
    
    print("Saving code ...")
    submission_files = [
        "data.py",
        "hourglass2D_model.py",
        "linear_model.py",
        "train_hourglass2D.py",
        "train_simple_baseline.py",
        "test_simple_baseline.py",
        "layers.py",
        "setup.py",
        "utils.py",
        "vis.py"
    ]
    utils.create_zip_code_files(os.path.join(submissions_dir, "code.zip"), submission_files)
    print("Done.")


def main(_):
    test()

if __name__ == "__main__":
  tf.app.run()
