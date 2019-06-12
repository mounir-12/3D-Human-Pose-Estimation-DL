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
import cv2

# fix random seeds in this file since used by all training files to load data
seed = 5
tf.random.set_random_seed(seed)
np.random.seed(seed)
print("\nFixed random seeds\n")

max_rotation = (np.pi/180) * 30 # 30 degrees in radians
max_scaling = 1.2
p_aug = 0.4

# ---------------------------------------- Basic Data Loading Function -------------------------------------------------------

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image,tf.float32) / 128. - 1
    image = tf.transpose(a=image, perm=[2, 0, 1]) # images are read in channels_last format, so convert to channels_first format
    return image

def load_and_preprocess_image_and_pose(path, pose):
    image = tf.read_file(path)
    image = preprocess_image(image)
    pose = tf.cast(pose,tf.float32)
    return image,pose

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    image = preprocess_image(image)
    return image
    
def load_and_preprocess_image_and_poses(path, pose2d, pose3d):
    image = tf.read_file(path)
    image = preprocess_image(image)
    pose2d = tf.cast(pose2d,tf.float32)
    pose3d = tf.cast(pose3d,tf.float32)
    return image,pose2d,pose3d
    
# ---> data augmentation functions

def preprocess_image_aug(image, angle):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image,tf.float32) / 128. - 1
    image = tf.contrib.image.rotate( image, angle, interpolation='BILINEAR')
    image = tf.transpose(a=image, perm=[2, 0, 1]) # images are read in channels_last format, so convert to channels_first format
    return image

def augment_pose2d(image, pose2d, angle):
    h = tf.cast(tf.shape(image)[1], tf.float32) # image height
    w = tf.cast(tf.shape(image)[2], tf.float32) # image width
    center = tf.stack([w/2.0, h/2.0], axis=0) # center pixel of the image
    pose2d = pose2d - center # center the pose on the center pixel in order to rotate around it
    M = tf.transpose(tf.stack([[tf.cos(-angle), -tf.sin(-angle)], 
                               [tf.sin(-angle), tf.cos(-angle)]])) # rotation matrix "tansposed" because the pose joint is a row vector (i,e also transposed)
                                                                   # and we rotate by -angle because the y axis grows down (whereas it 
                                                                   # grows up in the euclidian coordinate system)
    pose2d = tf.matmul(pose2d, M) # rotate each joint of the pose
    pose2d = pose2d + center # center pose back
    return pose2d

def augment_pose3d(pose3d, angle):
    M = tf.transpose(tf.stack([[tf.cos(-angle), -tf.sin(-angle), tf.constant(0.0)], 
                                   [tf.sin(-angle), tf.cos(-angle), tf.constant(0.0)],
                                   [tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)]])) # rotation matrix "tansposed" because the pose joint is a row vector (i,e also transposed)
                                                                                             # and we rotate by -angle because the y axis grows down (whereas it 
                                                                                             # grows up in the euclidian coordinate system)
    pose3d = tf.matmul(pose3d, M) # rotate each joint of the pose
    return pose3d

def load_and_preprocess_image_and_pose2d_aug(path, pose2d):
    image = tf.read_file(path)
    pose2d = tf.cast(pose2d,tf.float32)

    def augment2d(image, pose2d, augment=True):
        if augment:
            angle = tf.random.uniform(shape=[], minval=-max_rotation, maxval=max_rotation, seed=seed)
            image = preprocess_image_aug(image, angle)
            pose2d = augment_pose2d(image, pose2d, angle)
            return image, pose2d
        else:
            return preprocess_image(image), pose2d
        
    rand = tf.random.uniform(shape=[], minval=0, maxval=1, seed=seed)
    image, pose2d = tf.cond(rand < tf.constant(5.0), lambda: augment2d(image, pose2d, True), lambda: augment2d(image, pose2d, False))
    return image, pose2d
    
def load_and_preprocess_image_and_pose3d_aug(path, pose3d):
    image = tf.read_file(path)
    pose3d = tf.cast(pose3d,tf.float32)
    
    def augment3d(image, pose3d, augment=True):
        if augment:
            angle = tf.random.uniform(shape=[], minval=-max_rotation, maxval=max_rotation, seed=seed)
            image = preprocess_image_aug(image, angle)
            pose3d = augment_pose3d(pose3d, angle)
            return image, pose3d
        else:
            return preprocess_image(image), pose3d
        
    rand = tf.random.uniform(shape=[], minval=0, maxval=1, seed=seed)
    image, pose3d = tf.cond(rand < tf.constant(5.0), lambda: augment3d(image, pose3d, True), lambda: augment3d(image, pose3d, False))
    return image, pose3d
    
def load_and_preprocess_image_and_pose2d_3d_aug(path, pose2d, pose3d):
    image = tf.read_file(path)
    pose2d = tf.cast(pose2d, tf.float32)
    pose3d = tf.cast(pose3d, tf.float32)
    
    def augment2d_3d(image, pose2d, pose3d, augment=True):
        if augment:
            angle = tf.random.uniform(shape=[], minval=-max_rotation, maxval=max_rotation, seed=seed)
            image = preprocess_image_aug(image, angle)
            pose2d = augment_pose2d(image, pose2d, angle)
            pose3d = augment_pose3d(pose3d, angle)
            return image, pose2d, pose3d
        else:
            return preprocess_image(image), pose2d, pose3d
        
    rand = tf.random.uniform(shape=[], minval=0, maxval=1, seed=seed)
    image, pose2d, pose3d = tf.cond(rand < tf.constant(5.0), lambda: augment2d_3d(image, pose2d, pose3d, True), lambda: augment2d_3d(image, pose2d, pose3d, False))
    return image, pose2d, pose3d
# --->

def create_dataloader_train(data_root, batch_size, valid_subject=None, valid_size=None, batches_to_prefetch=1000, data_to_load="pose3d", shuffle=True, augment=True):
    assert(not (valid_subject is not None and valid_size is not None)) # we can choose to either validate on a subject or on a random validation set with some validation_size, not both
    print("\nCreating Dataset...")
    phase = "train"
    all_image_names = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines() # read all lines ('\n' included)
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_names] # construct paths removing '\n'

    annotations_path = os.path.join(data_root,"annot","%s.h5"%phase)
    annotations = h5py.File(annotations_path, 'r')

    if valid_subject is not None:
        valid_indices = set([i for i in range(len(all_image_names)) if all_image_names[i].split("_")[0] == valid_subject])
        print("Validation subject: ", valid_subject)
        print("Validation set size: ", len(valid_indices))
        min_valid_index, max_valid_index = min(valid_indices), max(valid_indices)
        # split dataset to train and validation
        train_image_paths = all_image_paths[:min_valid_index] + all_image_paths[max_valid_index+1:]
        train_pose2d = np.concatenate([annotations["pose2d"][:min_valid_index],annotations["pose2d"][max_valid_index+1:]])
        train_pose3d = np.concatenate([annotations["pose3d"][:min_valid_index],annotations["pose3d"][max_valid_index+1:]])
        
        valid_image_paths = all_image_paths[min_valid_index:max_valid_index+1]
        valid_pose2d = annotations["pose2d"][min_valid_index:max_valid_index+1]
        valid_pose3d = annotations["pose3d"][min_valid_index:max_valid_index+1]
    elif valid_size is not None:
        print("Validation set size: ", valid_size)
    else:
        print("No validation set created")
    
    if data_to_load == "all_poses":
        means_2d = np.mean(annotations["pose2d"], axis=0).flatten()
        std_2d = np.std(annotations["pose2d"], axis=0).flatten()
        means_3d = np.mean(annotations["pose3d"], axis=0).flatten()
        std_3d = np.std(annotations["pose3d"], axis=0).flatten()
        processing_func = load_and_preprocess_image_and_poses # function to call in the map operation below
        if augment:
            processing_func = load_and_preprocess_image_and_pose2d_3d_aug
        if valid_subject is not None: # create train and validation datasets
            train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_pose2d, train_pose3d))
            valid_ds = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_pose2d, valid_pose3d))
        else: # create only train dataset
            train_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, annotations["pose2d"], annotations["pose3d"]))
    else:
        means = np.mean(annotations[data_to_load], axis=0).flatten()
        std = np.std(annotations[data_to_load], axis=0).flatten()
        processing_func = load_and_preprocess_image_and_pose # function to call in the map operation below
        if augment:
            if data_to_load == "pose2d":
                processing_func = load_and_preprocess_image_and_pose2d_aug
            else:
                processing_func = load_and_preprocess_image_and_pose3d_aug # function to call in the map operation below
        if valid_subject is not None: # create train and validation datasets
            if data_to_load == "pose2d":
                train_pose, valid_pose = [train_pose2d, valid_pose2d]
            else:
                train_pose, valid_pose = [train_pose3d, valid_pose3d]
            
            train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_pose))
            valid_ds = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_pose))
        else: # create only train dataset
            train_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, annotations[data_to_load]))
    
    if shuffle:
        print("Shuffling data...")
        if valid_subject is not None: # shuffle both train and validation dataset
            train_ds = train_ds.shuffle(buffer_size=len(all_image_paths) - len(valid_indices))
            valid_ds = valid_ds.shuffle(buffer_size=len(valid_indices))
        else: # shuffle only train dataset
            train_ds = train_ds.shuffle(buffer_size=len(all_image_paths))

    print("Mapping Data...")
    train_ds = train_ds.map(processing_func)
    if valid_subject is not None:
        valid_ds = valid_ds.map(processing_func)

    if valid_size is not None: # if we chose to pick a random validation set with a mix of subjects, then we do the train/validation split here
        valid_ds = train_ds.take(valid_size)
        train_ds = train_ds.skip(valid_size)
    
    print("Batching Data...")
    train_ds = train_ds.repeat() # repeat dataset indefinitely
    train_ds = train_ds.batch(batch_size, drop_remainder=True) # batch data
    train_ds = train_ds.prefetch(batches_to_prefetch)
    
    train_ds = train_ds.make_one_shot_iterator().get_next() # convert to iterator
    
    if valid_subject is not None or valid_size is not None: # batch and prefetech validation data
        valid_ds = valid_ds.repeat() # repeat dataset indefinitely
        valid_ds = valid_ds.batch(batch_size, drop_remainder=True) # batch data
        valid_ds = valid_ds.prefetch(batches_to_prefetch)
        
        valid_ds = valid_ds.make_one_shot_iterator().get_next() # convert to iterator
    
    # decide what to return
    to_return = [train_ds] # list of objects to return
    
    if valid_subject is not None:
        to_return.append(valid_ds)
        to_return.append(len(valid_indices)) # also return size of validation data
    elif valid_size is not None:
        to_return.append(valid_ds)
        to_return.append(valid_size) # also return size of validation data
    
    if data_to_load == "all_poses":
        to_return = to_return + [means_2d, std_2d, means_3d, std_3d] # return 2d and 3d means and std
    else:
        to_return = to_return + [means, std] # return means and std of loaded poses
    
    print("Done ...\n")
    return to_return
        
def create_dataloader_test(data_root): # return a dataloader i,e an iterator
    print("Generating test dataset")
    phase = "valid"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]

    image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = image_ds.map(load_and_preprocess_image)
    image_ds = image_ds.repeat() # repeat forever

    image_ds = image_ds.batch(1, drop_remainder=True) # batch size is 1

    iterator = image_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    print("Done ...")
    return dataloader
    
# -------------------------------------------- Bones Resnet 50 Specific Functions -------------------------------------------------

def preprocess_image_resnet50(image, height, width):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (height, width))
    image = tf.cast(image,tf.float32) / 255.
    return image
    
def load_and_preprocess_image_and_pose_resnet50(path, pose):
    image = tf.read_file(path)
    height = 224
    width = 224
    image = preprocess_image_resnet50(image, height, width)
    pose = tf.cast(pose,tf.float32)
    return image,pose

def load_and_preprocess_image_renet50(path):
    image = tf.read_file(path)
    image = preprocess_image_resnet50(image)
    return image

def create_dataloader_train_resnet50(data_root, batch_size, batches_to_prefetch=1000, data_to_load="pose3d", shuffle=True):
    phase = "train"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines() # read all lines ('\n' included)
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths] # construct paths removing '\n'

    annotations_path = os.path.join(data_root,"annot","%s.h5"%phase)
    annotations = h5py.File(annotations_path, 'r')
    
    means = np.mean(annotations[data_to_load], axis=0).flatten()
    std = np.std(annotations[data_to_load], axis=0).flatten()
    
    image_pose_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, annotations[data_to_load])) # dataset of zipped paths and 3D poses (i,e tuples)
    if shuffle:
        image_pose_ds = image_pose_ds.shuffle(buffer_size=len(all_image_paths)) # shuffle
    image_pose_ds = image_pose_ds.map(load_and_preprocess_image_and_pose_resnet50) # load images

    image_pose_ds = image_pose_ds.repeat() # repeat dataset indefinitely
    # image_pose_ds = image_pose_ds.batch(batch_size, drop_remainder=True) # batch data
    image_pose_ds = image_pose_ds.batch(batch_size) # batch data
    image_pose_ds.prefetch(batches_to_prefetch)
    
    iterator = image_pose_ds.make_one_shot_iterator() # create iterator
    dataloader = iterator.get_next() # object to get the next element every time we run it in a session

    return dataloader, means, std


def create_dataloader_test_resnet50(data_root, batch_size):
    phase = "valid"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]

    image_pose_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_pose_ds = image_pose_ds.map(load_and_preprocess_image_renet50)

    image_pose_ds = image_pose_ds.batch(batch_size)

    iterator = image_pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader

