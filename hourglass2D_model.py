import tensorflow as tf
from layers import *
from PIL import Image
import os, time

class StackedHourglass:
    def __init__(self, nb_joins=17, nb_channels=256, nb_modules=1, hourglass_half_size=4, nb_stacks=1, sigma=1):
        self.dtype = tf.float32 # data type to use
        
#        self.images_shape = (3, 256, 256) # the shape of input image with channels first (here: 3*256*256)
        self.nb_joins = nb_joins # nb of joins to be predicted
        self.nb_channels = nb_channels # nb of channels in the hourglass (expected as input to hourglass and preserved inside it) 
        self.nb_modules = nb_modules # nb of residual modules stacked in 1 residual block (which is the basic block in the network)
        self.n = hourglass_half_size # nb of residual blocks applied till the center of the hourglass
        self.nb_stacks = nb_stacks # nb of hourglasses stacked in the network
        self.var = sigma**2 # the variance used in each of the height and width axes for the gaussian heatmap
        
#        self.batch_shape = list(self.images_shape).copy() # copy shape
#        self.batch_shape.insert(0, None) # prepend None for variable batch size
        
    def __call__(self, inp, training=False):
        a = time.time()
#        self.inp = tf.placeholder(self.dtype, shape=self.batch_shape) # the input batch of images
#        self.training = tf.placeholder(tf.bool) # boolean used in batch_normalization layer, is True when training and False when predicting
        
        self.inp = inp
        conv1 = conv_layer(inp, 64, (7, 7), (2, 2), (3, 3)) # image size goes from 256*256 to 128*128, 64 channels
        relu1 = relu_layer(batch_norm_layer(conv1, training)) # apply batch norm then relu
        res1 = residual_hg(relu1, 128, training) # produce 128 channels from 64 channels
        max_pool1 = max_pool_layer(res1, (2, 2), (2, 2)) # 2x2 pooling with stride 2x2 => Down-sampling: we half the image height and width
        
        res2 = residual_hg(max_pool1, 128, training)
        res3 = residual_hg(res2, self.nb_channels, training)
        
        inter = res3 # intermediate input of the intermediate hourglass
        outputs = [] # the outputs of the hourglasses
        
        for i in range(self.nb_stacks):
            hg_out = self.hourglass(inter, self.n, training) # push intermediate input through hourglass
            hg_out = residual_block_hg(hg_out, self.nb_channels, self.nb_modules, training) # apply residual block one last time on gh_out which is the result of the last addition in 
                                                                                            # the hourglass (see paper)
            round1 = linear_layer(hg_out, self.nb_channels, training) # perform the 1st round of the 2 consecutive conv rounds, perform in addition to that a batch norm and relu
            
            heatmaps = conv_layer(round1, self.nb_joins, (1, 1), (1, 1), (0, 0)) # perform a conv with 1x1 kernel, 1x1 stride and 0x0 padding to get the predicted heatmaps
            outputs.append(heatmaps) # store the predicted heatmaps
            
            if(i < self.nb_stacks-1): # if this isn't the last hourglass
                round2 = conv_layer(round1, self.nb_channels, (1, 1), (1, 1), (0, 0)) # perform the 2nd round of the 2 consecutive conv rounds
                transformed = conv_layer(heatmaps, self.nb_channels, (1, 1), (1, 1), (0, 0)) # transform the predictions back with a conv layer to prepare for add-op (by having compatible nb channels)
                inter = inter + round2 + transformed # sum-up tensors which gives us the input for the next hourglass
        
        self.outputs = outputs
        
        final_output = outputs[-1]
        self.final_output = final_output
        
        # since output_shape != input_shape (in paper, 256*256 input, 64*64 output), we compute the scaling factor along H and W
        self.scale_H = tf.cast(tf.shape(final_output)[2] / tf.shape(inp)[2], tf.float32) # scale along H
        self.scale_W = tf.cast(tf.shape(final_output)[3] / tf.shape(inp)[3], tf.float32) # scale along W
        
        batch_size = tf.shape(inp)[0]
        self.scale = tf.tile(tf.reshape(tf.stack([self.scale_W, self.scale_H]), [1,1,2]), [batch_size, self.nb_joins, 1]) # tensor to rescale 2d poses to match output resolution 
                                                                                                                          # (i,e the scaling from input to output)
        p2d_pred_scaled_down = tf.map_fn(self.heatmaps_2d_to_pose, final_output) # transform each set of heatmaps in the batch to a p2d
        
        p2d_pred = p2d_pred_scaled_down / self.scale # divide by scale to scale up joints positions to input resolution
        self.p2d_pred = p2d_pred
        
        print("\n\nBuilt the model in {} s\n\n".format(time.time()-a))

        return outputs, p2d_pred # return all heatmaps of all hourglasses and the p2d predicted from the final_output
            
        
        
    def hourglass(self, inp, n, training): # recursive definition of an hourglass, n=number of residual_blocks (cubes in paper) till the center of the hourglass
        up_branch = residual_block_hg(inp, self.nb_channels, self.nb_modules, training) # push input through residual block and get output
        
        low_branch = max_pool_layer(inp, (2, 2), (2, 2)) # 2x2 pooling of input with stride 2x2 => Down-sampling: we half the input's height and width
        low_branch = residual_block_hg(low_branch, self.nb_channels, self.nb_modules, training) # push through a residual block
        
        if n > 1: # while n > 1, we keep performing a recursive call
            low_branch = self.hourglass(low_branch, n-1, training) # recursive call
        else: # n == 1
            low_branch = residual_block_hg(low_branch, self.nb_channels, self.nb_modules, training) # push through a residual block
        
        # starting here, we start building the right half of the hourglass
        low_branch = residual_block_hg(low_branch, self.nb_channels, self.nb_modules, training) # push through a residual block
        low_branch = nearest_neighbor_up_sampling_layer(low_branch, (2, 2)) # Up-sample: double the image width and height
        
        return up_branch + low_branch
        
    def pose_to_2d_heatmap(self, p2d, H, W): # expects a batch of poses
        p2d = tf.reshape(p2d, [-1, self.nb_joins, 2])
        return tf.map_fn(lambda joints: self.joints_to_2d_heatmap(joints, H, W), p2d) 
        
    def joints_to_2d_heatmap(self, joints, H, W): # expects only 1 pose (i,e the 17 joints)
        joints = tf.reshape(joints, [-1, 2])
        return tf.map_fn(lambda joint: self.gaussian_2d(joint, self.var, H, W), joints)
    
    def gaussian_2d(self, mean, var, H, W):
        mu = mean # x-axis grows to the right, y-axis grows to the bottom (mu[1] is for x-axis, mu[2] is for y-axis)
        cov = [[ var,  0.0],[ 0.0,  var]]

        #Multivariate Normal distribution
        gaussian = tf.contrib.distributions.MultivariateNormalFullCovariance(
                   loc=mu,
                   covariance_matrix=cov)

        # Generate a mesh grid to plot the distributions
        X, Y = tf.meshgrid(tf.range(0.0, W, 1), tf.range(0.0, H, 1))
        idx = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y,[-1,1])], axis=1)
        z = gaussian.prob(mu) # this is 1/normalization_factor
        prob = tf.reshape(gaussian.prob(idx)/z, tf.shape(X)) # create unnormalized gaussian
        return prob
        
    def heatmaps_2d_to_pose(self, heatmaps): # expects 1 set of heatmaps (i,e not a batch, but 17 heatmaps, one per joint)
        return tf.map_fn(lambda a: tf.roll(tf.cast(tf.unravel_index(
                                            tf.argmax(tf.reshape(a, [-1]), output_type=tf.int32), tf.shape(a)),tf.float32),
                                    axis=0, shift=1)
                        , heatmaps) # for each heatmap array "a", flatten array then get argmax index in flattened array then convert flat_index back to 2d index using unravel_index then cast 2d 
                                    # index to float32, then swap the result (using tf.roll) since the result is (row, col), but we want (col, row)=(x, y)
                                                                                                                                       
        
    def compute_loss(self, p2d_gt, all_heatmaps_pred):
        # we need to scale the ground-truth joints positions to account for the scaling from input to output image
        p2d_gt_scaled = p2d_gt*self.scale # multiply each joint by the scale
        
        heatmaps_gt = self.pose_to_2d_heatmap(p2d_gt_scaled, tf.shape(self.final_output)[2], tf.shape(self.final_output)[3])  # create a 2d gaussian heatmap from the scaled ground truth with same 
        
        total_loss = 0
        for heatmaps_pred in all_heatmaps_pred:
            loss = tf.losses.mean_squared_error(heatmaps_gt, heatmaps_pred)
            total_loss += loss
        self.loss = total_loss
        
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#            print("\n\nloss:", sess.run(self.loss), "\n\n")
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#            heatmaps_gt_arr, inp_arr = sess.run([heatmaps_gt, self.inp])
#            image = ((inp_arr[0]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
#            img = Image.fromarray(image, "RGB")
#            img = img.resize(heatmaps_gt_arr[0,0].shape) # resize input image to have the shape of the heatmaps
#            if not os.path.exists("./sample"):
#                os.makedirs("./sample")
#            img.save("./sample/img.png")
#            for i,heatmap in enumerate(heatmaps_gt_arr[0]):
#                print("heatmap {}, min: {}, max: {}".format(i, heatmap.min(), heatmap.max()))
#                heatmap = (heatmap*255).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
#                img = Image.fromarray(heatmap)
#                img.save("./sample/img_{}.png".format(i))
                
        return total_loss
            
    def get_train_op(self, loss, learning_rate=0.001):
        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, self.global_step)
        return self.train_op, self.global_step
        

        

        
    
