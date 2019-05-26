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
import datetime, time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

def get_time():
    return time.time()
    
# ------------------------------ Transformation -------------------------------------------------
def get_rot_matrix_around_z(degrees):
    M = np.zeros([3,3])
    M[0][0] = math.cos(math.radians(degrees))
    M[0][1] = -math.sin(math.radians(degrees))
    M[1][0] = math.sin(math.radians(degrees))
    M[1][1] = math.cos(math.radians(degrees))
    M[2][2] = 1
    return M
    
def get_rot_matrix_around_y(degrees):
    M = np.zeros([3,3])
    M[0][0] = math.cos(math.radians(degrees))
    M[0][2] = math.sin(math.radians(degrees))
    M[2][0] = -math.sin(math.radians(degrees))
    M[2][2] = math.cos(math.radians(degrees))
    M[1][1] = 1
    return M

def transform_pose(pose, M): # M: transformation matrix
    transformed = np.empty(pose.shape)
    for i,joint in enumerate(pose):
        transformed[i] = np.matmul(M, joint)
    return transformed
# ---------------------------------------------------------------------------------------------

# ------------------------------------ Saving Predictions -------------------------------------
def save_p2d_image(image, p2d_gt, p2d_pred, dir_name, index, radius=3):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    image_p2d = image
    
    # simple visually distinct colors, used to color the position of the ground truth and predicted joint with the same color for each joint
    colors = [(255, 255, 255), (0, 0, 0), # from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
              (255, 0, 0), (0, 255, 0), (255, 225, 25), (0, 0, 255), (245, 130, 48), (145, 30, 180),  
              (240, 50, 230), (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40), 
              (255, 250, 200), (128, 0, 0), (0, 0, 128), (128, 128, 128)
             ]
             
    if p2d_gt is not None: # if ground truth available (e,g during training) them mark them
        for i, joint in enumerate(p2d_gt): # mark all 2D ground truth joints on the image
            joint = (int(joint[0]), int(joint[1]))
            image_p2d = cv2.circle(image_p2d, joint, radius, colors[i],-1) # put a circle marker at the position of the joint in the image
        
    for i, joint in enumerate(p2d_pred): # mark all predicted 2D joints on the image
        pt1 = (int(joint[0]) - radius , int(joint[1]) - radius) # 2D coordinates of rectangle corner
        pt2 = (int(joint[0]) + radius , int(joint[1]) + radius) # 2D coordinates of opposite rectangle corner
        image_p2d = cv2.rectangle(image_p2d, pt1, pt2, colors[i],-1) # put a rectangle/square marker centered at the position of the joint in the image
        
    img = Image.fromarray(image_p2d, "RGB")
    img.save(os.path.join(dir_name, "img_{}.png".format(index)))
    
def save_p3d_image(image, p3d_gt, p3d_pred, dir_name, index):
    dir_name = os.path.join(dir_name, "img_{}".format(index))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    fig = plt.figure(figsize=(25, 25))
    if p3d_gt is not None: # create subplot for p3d_gt
        ax_p3d_gt = fig.add_subplot(311, projection='3d')
        ax_p3d_pred = fig.add_subplot(312, projection='3d')
        ax_img = fig.add_subplot(313)
    else: # only create subplots for p3d_pred and image
        ax_p3d_pred = fig.add_subplot(211, projection='3d')
        ax_img = fig.add_subplot(212)
    
    ax_img.imshow(image)
    
    for degrees in range(0, 360, 90): # plot 3D pose from 4 sides
        M = get_rot_matrix_around_y(degrees)
        
        if p3d_gt is not None:
            # plot p3d_gt
            transformed = transform_pose(p3d_gt, M)
            plot_3D_pose(transformed, ax_p3d_gt, "Ground Truth")
        
        # plot p3d_pred
        transformed = transform_pose(p3d_pred, M)
        plot_3D_pose(transformed, ax_p3d_pred, "Predictions")
        
        # save figure
        fig_name = "fig_rot_{}.png".format(degrees)
        fig.savefig(os.path.join(dir_name, fig_name))
        
        # clear 3D pose axes
        if p3d_gt is not None:
            ax_p3d_gt.clear()
        ax_p3d_pred.clear()
    plt.close()
        
def plot_3D_pose(channels, ax, title):
    vals = np.reshape( channels, (17, -1) )
    I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])  # start points
    J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,0,0,0], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I)):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)] # each of x, y and z is a pair (x_start, x_end), (y_start, y_end), (z_start, z_end) with coordinates 
                                                                                 # of start and end joints 
        ax.plot(x, y, z, lw=2, c='r' if LR[i] else 'b') # plot the line connecting the start and end point using x, y and z with line width = 2 and color "r"="red" or "b"="blue"

    RADIUS = 1000 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2] # the root's coordinates
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    
    # remove axis labels and set aspect ratio to be equal across axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.view_init(elev=-90., azim=-90)
    ax.set_aspect('equal')
    ax.set_title(title)
# ----------------------------------------------------------------------------------------------
    
def compute_MPJPE(p3d_out,p3d_gt,p3d_std=None):

    p3d_out_17x3 = tf.reshape(p3d_out, [-1, 17, 3])
    p3d_gt_17x3  = tf.reshape(p3d_gt, [-1, 17, 3])
    
    mse = (p3d_out_17x3 - p3d_gt_17x3)
    if p3d_std is not None:
        mse = mse * p3d_std
    mse = mse ** 2
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

def generate_submission_3d(predictions, out_path):
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
    
def save_2d_pred(predictions, out_path):
    ids = np.arange(1, predictions.shape[0] + 1).reshape([-1, 1])

    predictions = np.hstack([ids, predictions])

    joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head',
              'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    header = ["Id"]

    for j in joints:
        header.append(j + "_x")
        header.append(j + "_y")

    header = ",".join(header)
    np.savetxt(out_path, predictions, delimiter=',', header=header, comments='')

def create_zip_code_files(output_file, submission_files):
    patoolib.create_archive(output_file, submission_files)
