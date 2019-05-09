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
import os
from tqdm import trange
from utils import compute_MPJPE,normalize_pose
from data import create_dataloader_train
import resnet_model

NUM_SAMPLES= 312188

# Hyper parameters
NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.003
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = 2000

# Path
LOG_PATH = "./log/example/"
DATA_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:
    # load image and GT 3d pose
    im, p3d_gt = create_dataloader_train(data_root=DATA_PATH, batch_size=BATCH_SIZE)

    # load mean and std
    p3d_mean = np.loadtxt(os.path.join(DATA_PATH,'annot',"mean.txt")).reshape([1,17,3]).astype(np.float32)
    p3d_std = np.loadtxt(os.path.join(DATA_PATH,'annot',"std.txt")).reshape([1,17,3]).astype(np.float32)

    p3d_std = tf.constant(p3d_std)
    p3d_mean = tf.constant(p3d_mean)

    p3d_std = tf.tile(p3d_std,[BATCH_SIZE,1,1])
    p3d_mean = tf.tile(p3d_mean,[BATCH_SIZE,1,1])

    # normalize 3d pose
    p3d_gt = normalize_pose(p3d_gt,p3d_mean,p3d_std)

    # define resnet model
    model = resnet_model.Model()

    # predict 3d pose
    p3d_out = model(im, training=True)

    # compute loss
    loss = tf.losses.absolute_difference(p3d_gt, p3d_out)

    learning_rate = tf.placeholder(tf.float32, shape=[])

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
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) )

            # vis
            if i % LOG_ITER_FREQ == 0:
                _, summary = sess.run([train_op, merged],feed_dict={learning_rate: LEARNING_RATE})
                train_writer.add_summary(summary, i)
            else:
                _, = sess.run([train_op],feed_dict={learning_rate: LEARNING_RATE})

            # save model
            if i % SAVE_ITER_FREQ == 0:
                saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=i)

    saver.save(sess,os.path.join(LOG_PATH,"model"),global_step=int(NUM_EPOCHS * NUM_SAMPLES / BATCH_SIZE))