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
from data import create_dataloader_train, create_dataloader_test
from hourglass3D_model import C2FStackedHourglass
import time
from PIL import Image

NUM_SAMPLES= 312188
NUM_SAMPLES_TEST = 10987

# Train parameters
NUM_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE = 2.5*10**(-4)
LOG_ITER_FREQ = 100
VALID_ITER_FREQ = 500
VALID_STEPS = 10 # number of validation batches to use to compute the mean validation loss and mpjpe
SAVE_ITER_FREQ = 2000
VALID_SUBJECT="S1"
TEST_EVERY_EPOCH = True

# Model parameters
Z_RES=[1, 2, 4, 64]
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
    train_ds, valid_ds, VALID_SIZE, _, _ = create_dataloader_train(data_root=DATA_PATH, batch_size=BATCH_SIZE, valid_subject=VALID_SUBJECT, 
                                                                   batches_to_prefetch=BATCHES_TO_PREFETCH, data_to_load=DATA_TO_LOAD, shuffle=SHUFFLE)
    NUM_SAMPLES = NUM_SAMPLES - VALID_SIZE # update NUM_SAMPLES
    im_train, p3d_gt_train = train_ds
    im_valid, p3d_gt_valid = valid_ds
    
#    sys.exit(0)

    print("Building model...\n")

    # define and build the model
    with tf.variable_scope("model", reuse=False):
        model_train = C2FStackedHourglass(z_res=Z_RES, sigma=SIGMA)
        all_heatmaps_pred_train, p3d_pred_train = model_train(im_train, True)
        
    with tf.variable_scope("model", reuse=True):
        model_valid = C2FStackedHourglass(z_res=Z_RES, sigma=SIGMA)
        all_heatmaps_pred_valid, p3d_pred_valid = model_valid(im_valid, False)
    
    # test data
    im_test = create_dataloader_test(data_root=DATA_PATH) # load test data with batch_size=1
    with tf.variable_scope("model", reuse=True):
        model_test = C2FStackedHourglass(z_res=Z_RES, sigma=SIGMA)
        all_heatmaps_pred_test, p3d_pred_test = model_test(im_test, False)
    
#    sys.exit(0)
    
    # compute loss
    print("Loss...")
    loss_train = model_train.compute_loss(p3d_gt_train, all_heatmaps_pred_train)
    loss_valid = model_valid.compute_loss(p3d_gt_valid, all_heatmaps_pred_valid)
#    sys.exit(0)

    # define trainer
    print("Train op...")
    train_op, global_step = model_train.get_train_op(loss_train, learning_rate=LEARNING_RATE)
#    sys.exit(0)

    print("MPJPE ...")
    mpjpe_train = utils.compute_MPJPE(p3d_pred_train,p3d_gt_train)
    mpjpe_valid = utils.compute_MPJPE(p3d_pred_valid,p3d_gt_valid)
    
    # visualization related

    print("Summaries ...")
    train_summary = tf.summary.merge([tf.summary.scalar("train_loss", loss_train),
                                     tf.summary.scalar("train_mpjpe", mpjpe_train)])
    
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
                        images, p3d_gt_vals, p3d_pred_vals, loss_val, mpjpe_val = sess.run([im_valid, p3d_gt_valid, p3d_pred_valid, 
                                                                                           loss_valid, mpjpe_valid]) # get valid loss and mpjpe on 1 batch
                        
                        index = np.random.randint(BATCH_SIZE) # pick a random index to visualize
                        image = ((images[index]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
                        save_dir = os.path.join(LOG_PATH, "valid_samples")
                        utils.save_p3d_image(image, p3d_gt_vals[index], p3d_pred_vals[index], save_dir, global_step_val+1) # save the a random image of the batch with its predicted pose
                    else:
                        loss_val, mpjpe_val = sess.run([loss_valid, mpjpe_valid]) # get valid loss and mpjpe on 1 batch
                    
                    valid_loss += loss_val / VALID_STEPS # add to the mean
                    valid_mpjpe += mpjpe_val / VALID_STEPS # add to the mean
                
                summary = sess.run(valid_summary, {valid_loss_pl: valid_loss, valid_mpjpe_pl: valid_mpjpe})
                writer.add_summary(summary, global_step_val+1)
                
            if (i+1) % LOG_ITER_FREQ == 0: # if it's time log train summaries
                _, summary = sess.run([train_op, train_summary]) # get train summaries
                global_step_val = sess.run(global_step)
                writer.add_summary(summary, global_step_val+1)
                
            else: # otherwise, just train
                _, = sess.run([train_op])

            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                global_step_val = sess.run(global_step) # get the global step value
                saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val+1)
                gc.collect() # free-up memory once model saved
            
            # we finished an epoch, we predict on test set
            if TEST_EVERY_EPOCH and (i+1) % 5 == 0:
                print("End of epoch, saving model ...")
                global_step_val = sess.run(global_step) # get the global step value
                saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val+1)
                gc.collect() # free-up memory once model saved
                print("Predicting on test set...")
                predictions = None
                with trange(NUM_SAMPLES_TEST) as t_test: # generate predictions for all images
                    for j in t_test:
                        p3d_out_value = sess.run([p3d_pred_test])

                        if predictions is None:
                            predictions = p3d_out_value
                        else:
                            predictions = np.concatenate([predictions,p3d_out_value],axis=0)

                predictions = predictions.reshape([-1, 51])
                print("\nPredictions shape:", predictions.shape)
                
                submissions_dir = os.path.join(LOG_PATH, "submissions")
                if not os.path.exists(submissions_dir):
                    os.makedirs(submissions_dir)
                utils.generate_submission_3d(predictions, os.path.join(submissions_dir, "submission_{}.csv.gz".format(global_step_val+1)))
                
                submission_files = [
                    "data.py",
                    "hourglass3D_model.py",
                    "test_hourglass3D.py",
                    "train_hourglass3D.py",
                    "layers.py",
                    "setup.py",
                    "utils.py",
                    "vis.py"
                ]
                utils.create_zip_code_files(os.path.join(submissions_dir, "code_{}.zip".format(global_step_val)), submission_files)
                    
    global_step_val = sess.run(global_step) # get the global step value
    saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val+1) # save at the end of training
