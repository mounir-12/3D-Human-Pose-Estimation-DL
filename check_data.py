import os
import h5py
from PIL import ImageFile

data_root = os.getcwd()

def get_data_shapes(phase):
    print("\n")
    all_image_paths = open(os.path.join(data_root,"annot","{}_images.txt".format(phase))).readlines() # read all lines ('\n' included)
    all_image_paths = [os.path.join(data_root, "images", path[:-1]) for path in all_image_paths] # construct paths removing '\n'
    
    if phase == "train":
        annotations_path = os.path.join(data_root,"annot","{}.h5".format(phase))
        annotations = h5py.File(annotations_path, 'r')
        
        available_data = [key for key in annotations.keys()]
        print("available", phase, "data", available_data)
    
    # Read all images shapes
    images_shapes = []
    for path in all_image_paths:
        with open(path, "rb") as f: # open image file
            shape = ()
            ImPar=ImageFile.Parser() # create image parser
            chunk = f.read(1024) # read a first chunk of image of size 1024 bytes = 1Kib
            while chunk != "": # keep reading while the read chunk is not empty
                ImPar.feed(chunk) # feed the chunk to image parser
                if ImPar.image: # if not None i,e if the previously read chunk are sufficient to construct the image object with its metadata
                    channels = 1
                    if(ImPar.image.mode == "RGB"):
                        channels = 3
                    img_size = ImPar.image.size
                    shape = (img_size[0], img_size[1], channels)
                    break
                chunk = f.read(1024) # otherwise read another chunk
            
            if shape not in images_shapes:
                images_shapes.append(shape)
    print(phase, "images shapes:", images_shapes, "and count:", len(all_image_paths))
    
    if phase == "valid":
        return
        
    # read all labels (i,e the 16 joints array) shapes
    labels_shapes_3d = []
    c = 0
    for label in annotations['pose3d']:
        c += 1
        if label.shape not in labels_shapes_3d:
            labels_shapes_3d.append(label.shape)
            
    print(phase, "labels shapes (3d):", labels_shapes_3d, "and count:", c)
    
    labels_shapes_2d = []
    c = 0
    for label in annotations['pose2d']:
        c += 1
        if label.shape not in labels_shapes_2d:
            labels_shapes_2d.append(label.shape)
            
    print(phase, "labels shapes (2d):", labels_shapes_2d, "and count:", c)
    
def print_root_joints(nb, data_to_read="pose2d"):
    annotations_path = os.path.join(data_root,"annot","train.h5")
    annotations = h5py.File(annotations_path, 'r')
    
    for i in range(nb):
        print(annotations[data_to_read][i][0])
    
#get_data_shapes("train")
#get_data_shapes("valid")

print_root_joints(20)


