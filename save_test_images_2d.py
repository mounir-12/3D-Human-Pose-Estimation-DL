import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from data import *
import utils
from PIL import Image

parser = ArgumentParser()

parser.add_argument("-fp", "--file_path", required=True, help="the predictions file path")
args = parser.parse_args()

DATA_PATH="."
LOG_PATH="Test_Images_Preds"
PRED_PATH=args.file_path

preds = np.genfromtxt(PRED_PATH, delimiter=",", skip_header=1)[:,1:].reshape([-1, 17, 2])

im_test = create_dataloader_test(data_root=DATA_PATH)
sess = tf.Session()

print("\nPreds shape: ", preds.shape, "\n")

for j in range(preds.shape[0]): # for each image
    print("image", j)                                                      
    image = sess.run(im_test)
    image = ((image[0]+1)*128.0).transpose(1,2,0).astype("uint8")
    save_dir = os.path.join(LOG_PATH, "test_samples_2d")
    image = np.asarray(Image.fromarray(image, "RGB")) # necessary conversion for cv2
    utils.save_p2d_image(image, None, preds[j], save_dir, j) # save the a random image of the batch with its predicted pose
                        
    
sess.close()
