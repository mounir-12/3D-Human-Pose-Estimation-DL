"""Copyright (c) 2019 AIT Lab, ETH Zurich, Xu Chen

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
import cv2

CLUSTER_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"
LOCAL_PATH = "."
if os.path.exists(CLUSTER_PATH):
    DATA_PATH = CLUSTER_PATH
else:
    DATA_PATH = LOCAL_PATH
    
max_num = 10 # max number of images to show with their 2D joints + 3D skeleton

def show3Dpose(channels, ax, nb_joints_to_show=17):

    vals = np.reshape( channels, (17, -1) )
    I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])  # start points
    J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,0,0,0], dtype=bool)

    # Make connection matrix
    for i in np.arange( nb_joints_to_show-1):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)] # each of x, y and z is a pair (x_start, x_end), (y_start, y_end), (z_start, z_end) with coordinates 
                                                                                 # of start and end joints 
        ax.plot(x, y, z, lw=2, c='r' if LR[i] else 'b') # plot the line connecting the start and end point using x, y and z with line width = 2 and color "r"="red" or "b"="blue"

    RADIUS = 750 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2] # the root's coordinates
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    ax.view_init(elev=-90., azim=-90)
    
    # remove axis labels and set aspect ratio to be equal across axes
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])
#    ax.set_zticklabels([])
    ax.set_aspect('equal')



all_image_paths = open(os.path.join(DATA_PATH,"annot","train_images.txt")).readlines()
all_image_paths = [os.path.join(DATA_PATH, "images", path[:-1]) for path in all_image_paths]

annotations_path = os.path.join(DATA_PATH,"annot","train.h5")
annotations = h5py.File(annotations_path, 'r')
np.set_printoptions(suppress=True)

for i,path in enumerate(all_image_paths):
    if i >= max_num: # only show the max_num images with their 2D joints + 3D skeleton
        break;
# Some interesting images for 3d pose:
#   image i = 223431 has max distance between any pair of joints along x-axis
#   image i = 141152 has max distance between any pair of joints along y-axis
#   image i = 152072 has max distance between any pair of joints along z-axis
#   image i = 98452 has max x-value for some joint in it
#    if i != 152072:
#        continue
    print(path)

# empirically, every pixel in 2d correspond to around 6.5 mm distance in 3d along x axis
# so given a joint j with x_j,2d in pixels and x_j,3d in mm, we can roughly convert as follows: x_j,3d = (x_j,2d - x_root,2d) * 6.5 (with some of around 7 pixels, i,e around 7*6.5mm since 6.5 is 
# just an estimate)

    fig = plt.figure()
    ax_img = fig.add_subplot(131)
    ax_p2d = fig.add_subplot(132)
    ax_p3d = fig.add_subplot(133, projection='3d') # the 3d axes, x-axis grows to the right, y-axis grows to the bottom, z-axis grows towards the screen, origin is at the pelvis (root joint)

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax_img.imshow(image)

    p2d = annotations["pose2d"][i] # get the i'th image's 2D joints
    image_p2d = image.copy()
    for joint_id in range( p2d.shape[0]): # mark all 2D joint on the image
        joint = ( int(p2d[joint_id,0]),int(p2d[joint_id,1])) # 2D coordinates of the joint
        image_p2d = cv2.circle(image_p2d, joint, 3, (0,255,0),-1) # put a circle marker at the position of the joint in the image
    ax_p2d.imshow(image_p2d)

    p3d = annotations["pose3d"][i]
    print(annotations["pose2d"][i])
    print(p3d)
    print(p3d.shape)
    show3Dpose(p3d, ax_p3d)
    plt.show()
