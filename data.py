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
import h5py
import os
import numpy as np
 
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image,tf.float32) / 128. - 1
    image = tf.transpose(a=image, perm=[2, 0, 1]) # images are read in channels_last format, so convert to channels_first format
    return image

def load_and_preprocess_image_and_pose(path,pose):
    image = tf.read_file(path)
    image = preprocess_image(image)
    pose = tf.cast(pose,tf.float32)
    return image,pose

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    image = preprocess_image(image)
    return image
    
def load_and_preprocess_image_and_poses(path,pose2d, pose3d):
    image = tf.read_file(path)
    image = preprocess_image(image)
    pose2d = tf.cast(pose2d,tf.float32)
    pose3d = tf.cast(pose3d,tf.float32)
    return image,pose2d,pose3d

def create_dataloader_train(data_root, batch_size, batches_to_prefetch=1000, data_to_load="pose3d", shuffle=True):
    phase = "train"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines() # read all lines ('\n' included)
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths] # construct paths removing '\n'

    annotations_path = os.path.join(data_root,"annot","%s.h5"%phase)
    annotations = h5py.File(annotations_path, 'r')
    
    if data_to_load == "all_poses":
        image_pose_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, annotations["pose2d"], annotations["pose3d"]))
        means_2d = np.mean(annotations["pose2d"], axis=0).flatten()
        std_2d = np.std(annotations["pose2d"], axis=0).flatten()
        means_3d = np.mean(annotations["pose3d"], axis=0).flatten()
        std_3d = np.std(annotations["pose3d"], axis=0).flatten()
        processing_func = load_and_preprocess_image_and_poses # function to call in the map operation below
    else:
        means = np.mean(annotations[data_to_load], axis=0).flatten()
        std = np.std(annotations[data_to_load], axis=0).flatten()
        image_pose_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, annotations[data_to_load])) # dataset of zipped paths and 3D poses (i,e tuples)
        processing_func = load_and_preprocess_image_and_pose # function to call in the map operation below
    
    if shuffle:
        image_pose_ds = image_pose_ds.shuffle(buffer_size=len(all_image_paths)) # shuffle
        
    image_pose_ds = image_pose_ds.map(processing_func) # load images

    image_pose_ds = image_pose_ds.repeat() # repeat dataset indefinitely
    image_pose_ds = image_pose_ds.batch(batch_size, drop_remainder=True) # batch data
    image_pose_ds.prefetch(batches_to_prefetch)
    
    iterator = image_pose_ds.make_one_shot_iterator() # create iterator
    dataloader = iterator.get_next() # object to get the next element every time we run it in a session
    
    if data_to_load == "all_poses":
        return dataloader, means_2d, std_2d, means_3d, std_3d
    else:
        return dataloader, means, std


def create_dataloader_test(data_root, batch_size):
    phase = "valid"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]

    image_pose_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_pose_ds = image_pose_ds.map(load_and_preprocess_image)

    image_pose_ds = image_pose_ds.batch(batch_size, drop_remainder=True)

    iterator = image_pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader







