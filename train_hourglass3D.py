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
import os, sys, gc
from tqdm import trange
import utils
from data import create_dataset_train
from hourglass3D_model import C2FStackedHourglass
import time
from PIL import Image

NUM_SAMPLES= 312188

# Train parameters
NUM_EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 2.5*10**(-4)
LOG_ITER_FREQ = 100
VALID_ITER_FREQ = 500
SAVE_ITER_FREQ = 2000
VALID_SUBJECT="S1"
VALID_STEPS = 10 # number of validation batches to use to compute the mean validation loss and mpjpe

# Model parameters
Z_RES=[1, 64]
SIGMA=2

# Data parameters
SHUFFLE=True
DATA_TO_LOAD="pose3d"
BATCHES_TO_PREFETCH=20

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
    train_ds, valid_ds, VALID_SIZE, _, _ = create_dataset_train(data_root=DATA_PATH, batch_size=BATCH_SIZE, valid_subject=VALID_SUBJECT, 
                                                               batches_to_prefetch=BATCHES_TO_PREFETCH, data_to_load=DATA_TO_LOAD, shuffle=SHUFFLE)
    NUM_SAMPLES = NUM_SAMPLES - VALID_SIZE # update NUM_SAMPLES
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

    print("Building model...\n")
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
    
    valid_loss_pl = tf.placeholder(dtype=tf.float32)
    valid_mpjpe_pl = tf.placeholder(dtype=tf.float32)
    valid_summary = tf.summary.merge([tf.summary.scalar("valid_loss", valid_loss_pl), 
                                      tf.summary.scalar("valid_mpjpe", valid_mpjpe_pl)])
    
    writer = tf.summary.FileWriter(CHECKPOINTS_PATH, sess.graph)

    # initialize
    tf.global_variables_initializer().run()

    # define model saver
    saver = tf.train.Saver(tf.global_variables())

    # training loop
    train_feed_dict={handle: train_handle, training: True} # feed dict for training
    valid_feed_dict={handle: valid_handle, training: False} # feed dict for validation
    print("Training start ...\n")
    with trange(int(NUM_EPOCHS * (NUM_SAMPLES // BATCH_SIZE))) as t:
        for i in t:

	        # display training status
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES # nb of epochs completed (e,g 1.5 => one epoch and a half)
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES # nb of images processed in current epoch
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) ) # update displayed info, iter_percent = percentage of completion of current iteration (i,e epoch)

            if (i+1) % VALID_ITER_FREQ == 0: # validation
                valid_loss = 0
                valid_mpjpe = 0
                global_step_val = sess.run(global_step) # get the global step value
                for j in range(VALID_STEPS):
                    if j == 0: # when j == 0 we also output an image for visualization
                        images, p3d_gt_vals, p3d_pred_vals, loss_val, mpjpe_val = sess.run([im, p3d_gt, p3d_pred, loss, mpjpe], valid_feed_dict) # get valid loss and mpjpe on 1 batch
                        
                        index = numpy.random.randint(BATCH_SIZE) # pick a random index to visualize
                        image = ((images[index]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
                        save_dir = os.path.join(LOG_PATH, "valid_samples")
                        utils.save_p3d_image(image, p3d_gt_vals[index], p3d_pred_vals[index], save_dir, global_step_val+1) # save the a random image of the batch with its predicted pose
                    else:
                        loss_val, mpjpe_val = sess.run([loss, mpjpe], valid_feed_dict) # get valid loss and mpjpe on 1 batch
                    
                    valid_loss += loss_val / VALID_STEPS # add to the mean
                    valid_mpjpe += mpjpe_val / VALID_STEPS # add to the mean
                
                summary = sess.run(valid_summary, {valid_loss_pl: valid_loss, valid_mpjpe_pl: valid_mpjpe})
                writer.add_summary(summary, global_step_val+1)
                
            if (i+1) % LOG_ITER_FREQ == 0: # if it's time log train summaries
                _, summary = sess.run([train_op, train_summary], train_feed_dict) # get train summaries
                global_step_val = sess.run(global_step)
                writer.add_summary(summary, global_step_val+1)
                
            else: # otherwise, just train
                _, = sess.run([train_op], train_feed_dict)

            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                global_step_val = sess.run(global_step) # get the global step value
                saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val+1)
                gc.collect() # free-up memory once model saved
            
            if (i+1) % (NUM_SAMPLES // BATCH_SIZE) == 0: # we finished an epoch
                print("Done an epoch")
                    
            
    saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE)) # save at the end of training
