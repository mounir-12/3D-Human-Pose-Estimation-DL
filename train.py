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
from utils import compute_MPJPE,normalize_pose_3d, normalize_pose_2d
from data import create_dataloader_train
import resnet_model
from hourglass2D_model import StackedHourglass
import time

NUM_SAMPLES= 312188

# Hyper parameters
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.003
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = 2000

# Path
LOG_PATH = "./log/example/"
CLUSTER_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"
LOCAL_PATH = "."
if os.path.exists(CLUSTER_PATH):
    DATA_PATH = CLUSTER_PATH
else:
    DATA_PATH = LOCAL_PATH


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:
    
    # load dataset of batched pairs (image, pose), means and stddev
    dataset, p2d_mean, p2d_std = create_dataloader_train(data_root=DATA_PATH, batch_size=BATCH_SIZE, data_to_load="pose2d", shuffle=False)
    im, p2d_gt = dataset # split the pairs (i,e unzip the tuple). When running one, the other also moves to the next elem (i,e same iterator)
        
    # mean and std
#    p2d_mean = p2d_mean.reshape([1,17,2]).astype(np.float32)
#    p2d_std = p2d_std.reshape([1,17,2]).astype(np.float32)

#    p2d_std = tf.constant(p2d_std)
#    p2d_mean = tf.constant(p2d_mean)

#    p2d_std = tf.tile(p2d_std,[BATCH_SIZE,1,1]) # repeat batch_size times along 0th dim
#    p2d_mean = tf.tile(p2d_mean,[BATCH_SIZE,1,1]) # repeat batch_size times along 0th dim

#    # normalize 2d pose
#    p2d_gt = normalize_pose_2d(p2d_gt,p2d_mean,p2d_std)

    # define model
    model = StackedHourglass()
    
    # build the model
    a = time.time()
    model(im, p2d_gt, training=True)
    
    print("time spent:", time.time()-a)
    sys.exit(0)
    print("success!!\n\n")

    # compute loss
    p2d_out, loss = tf.losses.absolute_difference(p3d_gt, p3d_out)

    # define trainer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm
    with tf.control_dependencies(update_ops):
        train_op = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE,momentum=0.9).minimize(loss)

    # compute MPJPE
    mpjpe = compute_MPJPE(p3d_out,p3d_gt,p3d_std)

    # visualization related
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("mpjpe", mpjpe)
    tf.summary.image("input", im, max_outputs=4)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

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
            if i % LOG_ITER_FREQ == 0:
                _, summary = sess.run([train_op, merged],feed_dict={learning_rate: LEARNING_RATE})
                train_writer.add_summary(summary, i)
            else:
                _, = sess.run([train_op],feed_dict={learning_rate: LEARNING_RATE})

            # save model
            if i % SAVE_ITER_FREQ == 0:
                saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=i)

    saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE)) # save at the end of training
