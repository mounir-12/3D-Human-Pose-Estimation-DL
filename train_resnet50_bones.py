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
import resnet_model
from hourglass2D_model import StackedHourglass
import time
from PIL import Image

NUM_SAMPLES= 312188

# Train parameters
NUM_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 0.03
LOG_ITER_FREQ = 50
SAVE_ITER_FREQ = 2000
SAMPLE_ITER_FREQ = 100

# Data parameters
SHUFFLE=True
DATA_TO_LOAD="pose3d"
BATCHES_TO_PREFETCH=300

# Paths
CURR_DIR = "."
LOG_PATH = os.path.join(CURR_DIR, "log", utils.timestamp())
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
	dataset, p3d_mean, p3d_std = create_dataloader_train(data_root=DATA_PATH, batch_size=BATCH_SIZE, batches_to_prefetch=BATCHES_TO_PREFETCH, data_to_load=DATA_TO_LOAD, shuffle=SHUFFLE)
    im, pose_gt = dataset # split the pairs (i,e unzip the tuple). When running one, the other also moves to the next elem (i,e same iterator)

    model = Resnet_50(nb_joints = 17, tfhub_module)
    pose_pred = model(im)

    loss = model.compute_loss(pose_gt, pose_pred)

    train_op, global_step = model.train_op(loss, learning_rate=LEARNING_RATE)

    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(CHECKPOINTS_PATH, sess.graph)

    saver = tf.train.Saver(tf.global_variables())

    with trange(int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE)) as t:
        for i in t:

	        # display training status
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES # nb of epochs completed (e,g 1.5 => one epoch and a half)
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES # nb of images processed in current epoch
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) ) # update displayed info, iter_percent = percentage of completion of current iteration (i,e epoch)

            # vis
            if (i+1) % SAMPLE_ITER_FREQ == 0: # if it's time to show sample images and predictions
                if (i+1) % LOG_ITER_FREQ == 0: # if it's also time to write summaries
                    _, images, pose_gt_arr, pose_pred_arr, summary = sess.run([train_op, im, pose_gt, pose_pred, merged])
                    train_writer.add_summary(summary, i+1) # write summary
                else: # otherwise, no summary writing
                    _, images, pose_gt_arr, pose_pred_arr = sess.run([train_op, im, pose_gt, pose_pred])

                image = ((images[0])*255.0).astype("uint8") # unnormalize and cast to uint8
                image = np.asarray(Image.fromarray(image, "RGB")) # necessary conversion for cv2
                save_dir = os.path.join(LOG_PATH, "train_samples")
                utils.save_p3d_image(image, p2d_gt_arr[0], p2d_pred_arr[0], save_dir, i+1)
            
            elif (i+1) % LOG_ITER_FREQ == 0:
                _, summary = sess.run([train_op, merged])
                train_writer.add_summary(summary, i+1)
            else:
                _, = sess.run([train_op])

            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model_resnet50"),global_step=i+1)

    saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE)) # save at the end of training