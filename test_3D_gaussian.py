import tensorflow as tf
from PIL import Image
import numpy as np
import os, shutil

tf.enable_eager_execution()
np.set_printoptions(threshold=np.inf)

def gaussian_3d(mean, var_x, var_y, var_z, H, W, D):
        mu = mean
        cov = [ [var_x, 0.0, 0.0], [0.0, var_y, 0.0], [0.0, 0.0, var_z]]

        #Multivariate Normal distribution
        gaussian = tf.contrib.distributions.MultivariateNormalFullCovariance(
                   loc=mu,
                   covariance_matrix=cov)

        # Generate a mesh grid to plot the distributions
        Y, Z, X = tf.meshgrid(tf.range(0.0, H, 1), tf.range(0.0, D, 1), tf.range(0.0, W, 1))
        idx = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y,[-1,1]), tf.reshape(Z,[-1,1])], axis=1)
        z = gaussian.prob(mu) # this is 1/normalization_factor
        prob = tf.reshape(gaussian.prob(idx)/z, [D, H, W]) # create unnormalized gaussian, channels_first
        return prob

var = 4.0
H = 64
W = 64
D_max = 64
D = 1
var_z = var*D/D_max
var_x = var_y = var
mean = [32,32,np.floor(32*D/D_max)] # take the floor() so that the mean_z falls in one of the of "depth planes"
prob = np.array(gaussian_3d(mean, var_x, var_y, var_z, H, W, D))

if not os.path.exists("3d_gaussian"):
    os.makedirs("3d_gaussian")
else:
    shutil.rmtree("3d_gaussian")
    os.makedirs("3d_gaussian")

for i,slice in enumerate(prob):
    print(slice.max())
    image = Image.fromarray((slice*255).astype("uint8")) # unnormalize
    image.save("3d_gaussian/img_{}.png".format(i))

