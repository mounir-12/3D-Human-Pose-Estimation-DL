"""Copyright (c) 2019 AIT Lab, ETH Zurich, Xu Chen

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import tensorflow as tf
import numpy as np
import os, sys
from tqdm import trange
import utils
from data import create_dataset_train
from hourglass3D_model import C2FStackedHourglass
import time
from PIL import Image

NUM_SAMPLES= 312188
VALID_SIZE= 2188

# Train parameters
NUM_EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 2.5*10**(-4)
LOG_ITER_FREQ = 100
VALID_ITER_FREQ = 50
SAVE_ITER_FREQ = 2000

# Model parameters
Z_RES=[64]
SIGMA=2

# Data parameters
SHUFFLE=True
DATA_TO_LOAD="pose3d"
BATCHES_TO_PREFETCH=300

# Paths
CURR_DIR = "."
LOG_PATH = os.path.join(CURR_DIR, "log_HG3D", utils.timestamp())
CHECKPOINTS_PATH = os.path.join(LOG_PATH, "checkpoints")
CLUSTER_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"
if os.path.exists(CLUSTER_PATH):
    DATA_PATH = CLUSTER_PATH
else:
    DATA_PATH = CURR_DIR

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:
    
    # load dataset of batched (image, pose2d, pose3d)
    train_ds, valid_ds, _, _ = create_dataset_train(data_root=DATA_PATH, batch_size=BATCH_SIZE, valid_size=VALID_SIZE, 
                                                    batches_to_prefetch=BATCHES_TO_PREFETCH, data_to_load=DATA_TO_LOAD, shuffle=SHUFFLE)
    # create a feedable/generic iterator (i,e an iterator to which we can feed any dataset's iterator)
    handle = tf.placeholder(tf.string, shape=[]) # handle (i,e string) used to refer to which dataset's iterator (train or valid) is to be used
    generic_iterator = tf.data.Iterator.from_string_handle(handle, train_ds.output_types, train_ds.output_shapes) # create generic_iterator that can refer to train_iterator or valid_iterator
                                                                                            # (depending on the handle fed to the placeholder, if fed the train_iterator_handle, then generic_iterator
                                                                                            # is the train_iterator but if fed the valid_iterator_handle then generic_iterator is valid_iterator)
    im, p3d_gt = generic_iterator.get_next() # split the data
    
    train_iterator = train_ds.make_one_shot_iterator() # create train iterator
    train_handle = sess.run(train_iterator.string_handle()) # get train_iterator_handle (i,e string id refering to train_iterator) 
                                                            # -> to be fed to "handle" when using training data (i,e when training)
    valid_iterator = valid_ds.make_one_shot_iterator() # create valid iterator
    valid_handle = sess.run(valid_iterator.string_handle()) # get valid_iterator_handle (i,e string id refering to the valid iterator)
                                                            # -> to be fed to "handle" when using validation data (i,e when validating)
    
#    sys.exit(0)

    # define model
    model = C2FStackedHourglass(z_res=Z_RES, sigma=SIGMA)
    
    # build the model
    training = tf.placeholder(tf.bool) # will be true when training and false when validating
    all_heatmaps_pred, p3d_pred = model(im, training)
#    sys.exit(0)
    
    # compute loss
    loss = model.compute_loss(p3d_gt, all_heatmaps_pred)
#    sys.exit(0)

    # define trainer
    train_op, global_step = model.get_train_op(loss, learning_rate=LEARNING_RATE)
#    sys.exit(0)

    mpjpe = utils.compute_MPJPE(p3d_pred,p3d_gt)
    
    # visualization related

    train_loss = tf.summary.scalar("train_loss", loss)
    train_mpjpe = tf.summary.scalar("train_mpjpe", mpjpe)
    train_summary = tf.summary.merge([train_loss, train_mpjpe])
    
    valid_loss = tf.summary.scalar("valid_loss", loss)
    valid_mpjpe = tf.summary.scalar("valid_mpjpe", mpjpe)
    valid_summary = tf.summary.merge([valid_loss, valid_mpjpe])
    
    writer = tf.summary.FileWriter(CHECKPOINTS_PATH, sess.graph)

    # initialize
    tf.global_variables_initializer().run()

    # define model saver
    saver = tf.train.Saver(tf.global_variables())

    # training loop
    train_feed_dict={handle: train_handle, training: True} # feed dict for training
    valid_feed_dict={handle: valid_handle, training: False} # feed dict for validation
    with trange(int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE)) as t:
        for i in t:

	        # display training status
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES # nb of epochs completed (e,g 1.5 => one epoch and a half)
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES # nb of images processed in current epoch
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) ) # update displayed info, iter_percent = percentage of completion of current iteration (i,e epoch)

            # vis
            if (i+1) % VALID_ITER_FREQ == 0:
                images, p3d_gt_vals, p3d_pred_vals, summary = sess.run([im, p3d_gt, p3d_pred, valid_summary], valid_feed_dict) # get valid summaries on 1 batch
                writer.add_summary(summary, i+1)
                
                image = ((images[0]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
                save_dir = os.path.join(LOG_PATH, "valid_samples")
                utils.save_p3d_image(image, p3d_gt_vals[0], p3d_pred_vals[0], save_dir, i+1) # save the 1st image of the batch with its predicted pose
                    
            if (i+1) % LOG_ITER_FREQ == 0: # if it's time log train summaries
                _, summary = sess.run([train_op, train_summary], train_feed_dict) # get train summaries
                writer.add_summary(summary, i+1)
                
            else: # otherwise, just train
                _, = sess.run([train_op], train_feed_dict)

            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=i+1)

    saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE)) # save at the end of training
