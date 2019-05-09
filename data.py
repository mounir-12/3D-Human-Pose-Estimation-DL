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


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image,tf.float32) / 128. - 1
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

def create_dataloader_train(data_root, batch_size):
    phase = "train"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]

    annotations_path = os.path.join(data_root,"annot","%s.h5"%phase)
    annotations = h5py.File(annotations_path, 'r')

    image_pose_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, annotations['pose3d']))
    image_pose_ds = image_pose_ds.shuffle(buffer_size=len(all_image_paths))
    image_pose_ds = image_pose_ds.map(load_and_preprocess_image_and_pose)

    image_pose_ds = image_pose_ds.repeat()
    image_pose_ds = image_pose_ds.batch(batch_size, drop_remainder=True)

    iterator = image_pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader


def create_dataloader_test(data_root, batch_size):
    phase = "valid"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]

    image_pose_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_pose_ds = image_pose_ds.map(load_and_preprocess_image)

    image_pose_ds = image_pose_ds.batch(batch_size)

    iterator = image_pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader
