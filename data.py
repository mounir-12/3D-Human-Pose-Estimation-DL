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

# fix random seeds in this file since used by all training files to load data
tf.random.set_random_seed(5)
np.random.seed(5)
print("\nFixed random seeds\n")

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

def create_dataset_train(data_root, batch_size, valid_subject=None, batches_to_prefetch=1000, data_to_load="pose3d", shuffle=True): # returns a dataset i,e not an iterator
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
    else:
        print("No validation set created")
    
    if data_to_load == "all_poses":
        means_2d = np.mean(annotations["pose2d"], axis=0).flatten()
        std_2d = np.std(annotations["pose2d"], axis=0).flatten()
        means_3d = np.mean(annotations["pose3d"], axis=0).flatten()
        std_3d = np.std(annotations["pose3d"], axis=0).flatten()
        processing_func = load_and_preprocess_image_and_poses # function to call in the map operation below
        if valid_subject is not None: # create train and validation datasets
            train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_pose2d, train_pose3d))
            valid_ds = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_pose2d, valid_pose3d))
        else: # create only train dataset
            train_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, annotations["pose2d"], annotations["pose3d"]))
    else:
        means = np.mean(annotations[data_to_load], axis=0).flatten()
        std = np.std(annotations[data_to_load], axis=0).flatten()
        processing_func = load_and_preprocess_image_and_pose # function to call in the map operation below
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

    print("Batching Data...")
    train_ds = train_ds.repeat() # repeat dataset indefinitely
    train_ds = train_ds.batch(batch_size, drop_remainder=True) # batch data
    train_ds = train_ds.prefetch(batches_to_prefetch)
    
    if valid_subject is not None: # batch and prefetech validation data
        valid_ds = valid_ds.repeat() # repeat dataset indefinitely
        valid_ds = valid_ds.batch(batch_size, drop_remainder=True) # batch data
        valid_ds = valid_ds.prefetch(batches_to_prefetch)

    # decide what to return
    to_return = [train_ds] # list of objects to return
    
    if valid_subject is not None:
        to_return.append(valid_ds)
        to_return.append(len(valid_indices)) # also return size of validation data
    
    if data_to_load == "all_poses":
        to_return = to_return + [means_2d, std_2d, means_3d, std_3d] # return 2d and 3d means and std
    else:
        to_return = to_return + [means, std] # return means and std of loaded poses
    
    print("Done ...\n")
    return to_return
        
def create_dataloader_test(data_root): # return a dataloader i,e an iterator
    phase = "valid"
    all_image_paths = open(os.path.join(data_root,"annot","%s_images.txt"%phase)).readlines()
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths]

    image_pose_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_pose_ds = image_pose_ds.map(load_and_preprocess_image)

    image_pose_ds = image_pose_ds.batch(1, drop_remainder=True) # batch size is 1

    iterator = image_pose_ds.make_one_shot_iterator()
    dataloader = iterator.get_next()

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
