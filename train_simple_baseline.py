"""Predicting 3d poses from 2d joints"""

import os
import sys
import time

import numpy as np
import tensorflow as tf

import linear_model
import utils
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--epochs', type = int, default = 200, help = 'number of training epochs')
parser.add_argument('--batch_size', type = int, default = 64, help = 'size of training batch')
parser.add_argument('--learning_rate', type = float, default = 1e-3, help = 'learning rate for the optimizer')

# Architecture
parser.add_argument('--linear_size', type = int, default = 1024, help = 'size of each model layer')
parser.add_argument('--num_layers', type = int, default = 2, help = 'number of layers in the model')
parser.add_argument('--no_residual', help = 'whether to add a residual connection every 2 layers', action="store_true") # by default, we use residual
parser.add_argument('--no_max_norm', help = 'whether to apply maxnorm constraint to the weights', action="store_true") # by default, we use max_norm
parser.add_argument('--no_batch_norm', help = 'whether to use batch_normalization', action="store_true") # by default, we use batch_norm
parser.add_argument('--dropout', type = float, default = 0.5, help = 'dropout keep probability. 1 means no dropout')

parser.add_argument('--train_dir', type = str, default = os.path.join(".", "log_SB", utils.timestamp()) , help = 'training directory')

# tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
# tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
# tf.app.flags.DEFINE_integer("epochs", 200, "How many epochs we should train for")

# # Architecture
# tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
# tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
# tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")
# tf.app.flags.DEFINE_boolean("max_norm", True, "Apply maxnorm constraint to the weights")
# tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")
# tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability. 1 means no dropout")

# # Directories
# tf.app.flags.DEFINE_string("train_dir", os.path.join(".", "log_SB", utils.timestamp()), "Training directory.")

# FLAGS = tf.app.flags.FLAGS

FLAGS = parser.parse_args()

FLAGS.residual = not FLAGS.no_residual
FLAGS.max_norm = not FLAGS.no_max_norm
FLAGS.batch_norm = not FLAGS.no_batch_norm

train_dir = FLAGS.train_dir

print("\n\n")
print("epochs:", FLAGS.epochs)
print("batch_size:", FLAGS.batch_size)
print("learning_rate:", FLAGS.learning_rate)
print("linear_size:", FLAGS.linear_size)
print("num_layers:", FLAGS.num_layers)
print("residual:", FLAGS.residual)
print("max_norm:", FLAGS.max_norm)
print("batch_norm:", FLAGS.batch_norm)
print("dropout:", FLAGS.dropout)
print("train_dir:", FLAGS.train_dir)
print("\n\n")
#sys.exit(0)

print("\n\nTrain dir: {}\n\n".format(train_dir))
summaries_dir = os.path.join( train_dir, "checkpoints" ) # Directory for TB summaries

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
      dtype=tf.float32)
  print("\n\nCreating model with fresh parameters.")
  session.run( tf.global_variables_initializer() )
  return model

def train():
  """Train a linear model for 3d pose estimation"""

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = "0"
  with tf.Session(config=config) as sess:

    # === Create the model ===
    print("\n\nCreating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model( sess, FLAGS.batch_size )
    model.train_writer.add_graph( sess.graph )
    print("Model created")

    #=== This is the training loop ===
    step_time, loss = 0.0, 0.0
    current_step = 0

    step_time, loss = 0, 0
    current_epoch = 0
    log_every_n_batches = 100

    for _ in range( FLAGS.epochs ):
      current_epoch = current_epoch + 1

      # === Load training batches for one epoch ===
      encoder_inputs, decoder_outputs = model.get_all_batches(training=True )
      nbatches = len( encoder_inputs )
      print("There are {0} train batches".format( nbatches ))
      start_time, loss = time.time(), 0.

      # === Loop through all the training batches ===
      for i in range( nbatches ):

        if (i+1) % log_every_n_batches == 0:
          # Print progress every log_every_n_batches batches
          print("Working on epoch {0}, batch {1} / {2}... ".format( current_epoch, i+1, nbatches), end="" )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        step_loss, loss_summary, lr_summary, _ =  model.step( sess, enc_in, dec_out, FLAGS.dropout, isTraining=True )

        if (i+1) % log_every_n_batches == 0:
          # Log and print progress every log_every_n_batches batches
          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / log_every_n_batches ) )

        loss += step_loss
        current_step += 1
        # === end looping through training batches ===

      loss = loss / nbatches
      print("=============================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (model.global_step.eval(),
            model.learning_rate.eval(), loss) )
      # === End training for an epoch ===

      # Save the model every epoch
      print( "Saving the model... ", end="" )
      start_time = time.time()
      model.saver.save(sess, os.path.join(summaries_dir, "model"), global_step=current_step )
      print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) )

      # Reset global time and loss
      step_time, loss = 0, 0

      sys.stdout.flush()

def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()
