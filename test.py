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
import resnet_model
from tqdm import trange
from data import create_dataloader_test
from utils import unnormalize_pose, generate_submission,create_zip_code_files
import numpy as np
import os
import math

# Global Attribute
BATCH_SIZE = 32
LOG_PATH  ="./log/example"
DATA_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"

with tf.Session() as sess:
    # load image and GT 3d pose
    im = create_dataloader_test(data_root=DATA_PATH, batch_size=BATCH_SIZE)

    # normalize 3d pose
    p3d_mean = np.loadtxt(os.path.join(DATA_PATH, 'annot', "mean.txt")).reshape([1, 17, 3]).astype(np.float32)
    p3d_std = np.loadtxt(os.path.join(DATA_PATH, 'annot', "std.txt")).reshape([1, 17, 3]).astype(np.float32)

    p3d_std = tf.constant(p3d_std)
    p3d_mean = tf.constant(p3d_mean)

    p3d_std = tf.tile(p3d_std,[BATCH_SIZE,1,1])
    p3d_mean = tf.tile(p3d_mean,[BATCH_SIZE,1,1])

    # define resnet model
    model = resnet_model.Model()

    # predict 3d pose
    p3d_out = model(im, training=False)

    # compute MPJPE
    p3d_out = unnormalize_pose(p3d_out, p3d_mean, p3d_std)

    p3d_out = tf.cast(p3d_out,tf.int16)
    # restore weights
    saver = tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint(LOG_PATH))

    predictions = None
    with trange(math.ceil(10987/BATCH_SIZE)) as t:
        for i in t:
            p3d_out_value = sess.run(p3d_out)

            if predictions is None:
                predictions = p3d_out_value
            else:
                predictions = np.concatenate([predictions,p3d_out_value],axis=0)

    generate_submission(predictions, "submission.csv.gz")
    create_zip_code_files("code.zip")
