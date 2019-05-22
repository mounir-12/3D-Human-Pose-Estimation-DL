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
from data import create_dataloader_train
from hourglass3D_model import C2FStackedHourglass
import time
from PIL import Image

NUM_SAMPLES= 312188

# Train parameters
NUM_EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 0.001
LOG_ITER_FREQ = 100
SAMPLE_ITER_FREQ = 1000
SAVE_ITER_FREQ = 2000

# Model parameters
Z_RES=[1, 2, 4, 64]
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
    dataset, _, _ = create_dataloader_train(data_root=DATA_PATH, batch_size=BATCH_SIZE, batches_to_prefetch=BATCHES_TO_PREFETCH, data_to_load=DATA_TO_LOAD, shuffle=SHUFFLE)
    im, p3d_gt = dataset # split the pairs (i,e unzip the tuple). When running one, the other also moves to the next elem (i,e same iterator)

#    sys.exit(0)
    # define model
    model = C2FStackedHourglass(z_res=Z_RES, sigma=SIGMA)
    
    # build the model
    all_heatmaps_pred, p3d_pred = model(im, training=True)
#    sys.exit(0)
    
    # compute loss
    loss = model.compute_loss(p3d_gt, all_heatmaps_pred)
#    sys.exit(0)

    # define trainer
    train_op, global_step = model.get_train_op(loss, learning_rate=LEARNING_RATE)
#    sys.exit(0)

    mpjpe = utils.compute_MPJPE(p3d_pred,p3d_gt)
    # visualization related
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("mpjpe", mpjpe)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(CHECKPOINTS_PATH, sess.graph)

    # initialize
    tf.global_variables_initializer().run()

    # define model saver
    saver = tf.train.Saver(tf.global_variables())

    # training loop
    with trange(int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE)) as t:
        for i in t:

	        # display training status
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES # nb of epochs completed (e,g 1.5 => one epoch and a half)
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES # nb of images processed in current epoch
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) ) # update displayed info, iter_percent = percentage of completion of current iteration (i,e epoch)

            # vis
            if (i+1) % SAMPLE_ITER_FREQ == 0: # if it's time to show sample images and predictions
                if (i+1) % LOG_ITER_FREQ == 0: # if it's also time to write summaries
                    _, images, p3d_gt_vals, p3d_pred_vals, summary = sess.run([train_op, im, p3d_gt, p3d_pred, merged])
                    train_writer.add_summary(summary, i+1) # write summary
                else: # otherwise, no summary writing
                    _, images, p3d_gt_vals, p3d_pred_vals = sess.run([train_op, im, p3d_gt, p3d_pred])

                image = ((images[0]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
                save_dir = os.path.join(LOG_PATH, "train_samples")
                utils.save_p3d_image(image, p3d_gt_vals[0], p3d_pred_vals[0], save_dir, i+1)
            
            elif (i+1) % LOG_ITER_FREQ == 0:
                _, summary = sess.run([train_op, merged])
                train_writer.add_summary(summary, i+1)
                
            else:
                _, = sess.run([train_op])

            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=i+1)

    saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE)) # save at the end of training