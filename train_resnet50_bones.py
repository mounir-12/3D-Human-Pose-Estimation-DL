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
BATCH_SIZE = 4
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
    im, p3d_gt = dataset # split the pairs (i,e unzip the tuple). When running one, the other also moves to the next elem (i,e same iterator)

    model = Resnet_50(nb_joints = 17)
    model.build_graph()