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
from hourglass3D_model import C2FStackedHourglass
from PIL import Image

NUM_SAMPLES = 10987

# Model parameters
Z_RES=[1, 2, 4, 64]

# Path
list_of_files = glob.glob('./log_HG3D/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
RESTORE_PATH  =os.path.join(latest_dir, "checkpoints") # we restore the lastest saved model from the latest experiment
CLUSTER_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"
LOCAL_PATH = "."
if os.path.exists(CLUSTER_PATH):
    DATA_PATH = CLUSTER_PATH
else:
    DATA_PATH = LOCAL_PATH

with tf.Session() as sess:
    # load images
    im = create_dataloader_test(data_root=DATA_PATH) # load test data with batch_size=1

    # define model
    model = C2FStackedHourglass(z_res=Z_RES)
    
    # build the model
    all_heatmaps_pred, p3d_pred = model(im, training=False)

    # restore weights
    print("Restoring latest model from {}\n".format(RESTORE_PATH))
    saver = tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint(RESTORE_PATH))

    predictions = None
    with trange(NUM_SAMPLES) as t: # generate predictions for all images
        for i in t:
            image, p3d_out_value = sess.run([im, p3d_pred])
            
#            image = ((image[0]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
#            save_dir = os.path.join(os.getcwd(), "test_samples")
#            utils.save_p3d_image(image, None, p3d_out_value[0], save_dir, i+1)

            if predictions is None:
                predictions = p3d_out_value
            else:
                predictions = np.concatenate([predictions,p3d_out_value],axis=0)

    predictions = predictions.reshape([-1, 51])
    print(predictions.shape)
    utils.generate_submission_3d(predictions, "submission.csv.gz")
    submission_files = [
        "data.py",
        "hourglass3D_model",
        "test_hourglass3D.py",
        "train_hourglass3D.py",
        "layers.py",
        "setup.py",
        "utils.py",
        "vis.py"
    ]
    utils.create_zip_code_files("code.zip", submission_files)




