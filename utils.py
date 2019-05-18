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
import patoolib, os, cv2
from PIL import Image

def convert_heatmaps_to_p2d(heatmaps):
    p2d = np.empty([heatmaps.shape[0], 2])
    for i, heatmap in enumerate(heatmaps):
        p2d[i] = np.unravel_index(heatmap.argmax(), heatmap.shape)
    return p2d

def save_p2d_image(image, p2d, dir_name, i):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    image_p2d = image
    for joint_id in range( p2d.shape[0]): # mark all 2D joint on the image
        joint = ( int(p2d[joint_id,0]),int(p2d[joint_id,1])) # 2D coordinates of the joint
        image_p2d = cv2.circle(image_p2d, joint, 1, (0,255,0),-1) # put a circle marker at the position of the joint in the image
    img = Image.fromarray(image, "RGB")
    img.save(os.path.join(dir_name, "img{}.png".format(i)))
    
def compute_MPJPE(p3d_out,p3d_gt,p3d_std):

    p3d_out_17x3 = tf.reshape(p3d_out, [-1, 17, 3])
    p3d_gt_17x3  = tf.reshape(p3d_gt, [-1, 17, 3])

    mse = ( (p3d_out_17x3 - p3d_gt_17x3) * p3d_std) ** 2
    mse = tf.reduce_sum(mse, axis=2)
    mpjpe = tf.reduce_mean(tf.sqrt(mse))

    return mpjpe

def normalize_pose_3d(p3d,p3d_mean, p3d_std):
    # p3d of dimension [batch_size, nb_joints, nb_coordinates]
    root = tf.tile( tf.expand_dims(p3d[:,0,:],axis=1), [1,17,1]) # extracting the root joints via p3d[:,0,:] reduces the dimension by 1,
                                                                 # tf.expand_dims() just reshapes the result to put back this dimension
                                                                 # tf.tile() to replicate the root's coordinates j times j=#joints=17
    p3d = p3d - root

    p3d = (p3d-p3d_mean) / p3d_std
    p3d = tf.reshape(p3d, [-1, 51])
    return p3d
    
def normalize_pose_2d(p2d, p2d_mean, p2d_std):
    root = tf.tile( tf.expand_dims(p2d[:,0,:],axis=1), [1,17,1])
    p2d = p2d - root
    p2d = (p2d - p2d_mean) / p2d_std
    p2d = tf.reshape(p2d, [-1, 34])
    return p2d

def unnormalize_pose_3d(p3d,p3d_mean, p3d_std):

    b = tf.shape(p3d)[0] # batch_size

    p2d_17x2 = tf.reshape(p3d, [-1, 17, 2])
    root = p2d_17x2[:,0,:]
    root = tf.expand_dims(root,axis=1)
    root = tf.tile(root,[1,17,1])
    p3d_17x3 = p3d_17x3 - root
    p3d_17x3 = p3d_17x3 * p3d_std[:b,...] + p3d_mean[:b,...]
    p3d = tf.reshape(p3d_17x3, [-1,51])
    return p3d
    
def unnormalize_pose_3d(p3d,p3d_mean, p3d_std):

    b = tf.shape(p3d)[0]

    p3d_17x3 = tf.reshape(p3d, [-1, 17, 3])
    root = p3d_17x3[:,0,:]
    root = tf.expand_dims(root,axis=1)
    root = tf.tile(root,[1,17,1])
    p3d_17x3 = p3d_17x3 - root
    p3d_17x3 = p3d_17x3 * p3d_std[:b,...] + p3d_mean[:b,...]
    p3d = tf.reshape(p3d_17x3, [-1,51])
    return p3d

def generate_submission(predictions, out_path):
    ids = np.arange(1, predictions.shape[0] + 1).reshape([-1, 1])

    predictions = np.hstack([ids, predictions])

    joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head',
              'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    header = ["Id"]

    for j in joints:
        header.append(j + "_x")
        header.append(j + "_y")
        header.append(j + "_z")

    header = ",".join(header)
    np.savetxt(out_path, predictions, delimiter=',', header=header, comments='')


submission_files = [
    "data.py",
    "resnet_model.py",
    "test.py",
    "train.py",
    "utils.py",
    "vis.py"
]

def create_zip_code_files(output_file):
    patoolib.create_archive(output_file, submission_files)
