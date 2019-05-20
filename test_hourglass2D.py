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
import utils
import numpy as np
import os, glob, sys
import math
from hourglass2D_model import StackedHourglass
from PIL import Image

# Model parameters
NB_STACKS=4
SIGMA=1

# Global Attribute
BATCH_SIZE = 1 # predict one by one, allows to use drop_remainder=true which gives back a batched set with a know batch dimension (used by tf.unstack() in HG model)

# Path
list_of_files = glob.glob('./log/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
RESTORE_PATH  =os.path.join(latest_dir, "checkpoints") # we restore the lastest saved model from the latest experiment
CLUSTER_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"
LOCAL_PATH = "."
if os.path.exists(CLUSTER_PATH):
    DATA_PATH = CLUSTER_PATH
else:
    DATA_PATH = LOCAL_PATH

with tf.Session() as sess:
    # load image and GT 2d pose
    im = create_dataloader_test(data_root=DATA_PATH, batch_size=BATCH_SIZE)

    print("\n", im)
    # define model
    model = StackedHourglass(nb_stacks=NB_STACKS, sigma=SIGMA)
    
    # build the model
    all_heatmaps_pred, p2d_pred = model(im, training=False)

    # restore weights
    saver = tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint(RESTORE_PATH))

    predictions = None
    with trange(math.ceil(10987/BATCH_SIZE)) as t: # generate predictions for all images
        for i in t:
            image, p2d_out_value = sess.run([im, p2d_pred])
            
#            image = ((image[0]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
#            image = np.asarray(Image.fromarray(image, "RGB")) # necessary conversion for cv2
#            save_dir = os.path.join(os.getcwd(), "test_samples")
#            utils.save_p2d_image(image, None, p2d_out_value[0], save_dir, i+1, radius=3)

            if predictions is None:
                predictions = p2d_out_value
            else:
                predictions = np.concatenate([predictions,p2d_out_value],axis=0)

    predictions = predictions.reshape([-1, 34])
    utils.save_2d_pred(predictions, "p2d_test.csv")



