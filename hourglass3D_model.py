import tensorflow as tf
from layers import *
from PIL import Image
import os, time

class C2FStackedHourglass: # Coarse to Fine Stacked Hourglass
    def __init__(self, nb_joints=17, input_shape=[3,256,256], output_shape=[64,64,64], nb_channels=512, hourglass_half_size=4, z_res=[1,2,4,64], sigma=2):
        self.dtype = tf.float32 # data type to use
        
        assert(output_shape[0] == z_res[-1]) # the depth resolutions must be the same
        
        self.input_shape = input_shape # the shape of input image with channels first (here: 3*256*256)
        self.output_shape = output_shape # the shape of the 3D heatmap of each joint at the end of the network, given as [depth, height, width]
        
        self.nb_joints = nb_joints # nb of joints to be predicted
        self.z_res = z_res # the resolution of the 3D output (and gaussian heatmap) along z-axis for each joint (z_res[i] for hourglass i)

        self.nb_channels = nb_channels # nb output channels of each hourglass
        self.n = hourglass_half_size # nb of residual blocks applied till the center of the hourglass
        self.nb_stacks = len(z_res) # nb of hourglasses stacked in the network
        
        self.var = sigma**2 # the variance used in each of the height, width and depth axes for the gaussian heatmap
        
        self.shift_x = 1000
        self.shift_y = 1000
        self.shift_z = 1000
        
        self.range_x = [0, 2000]
        self.range_y = [0, 2000]
        self.range_z = [0, 2000]

    def __call__(self, inp, training=False):
        a = time.time()
        
        # initial processing of the image
        self.inp = inp
        with tf.name_scope("preprocessing_input"):
            conv1 = conv_layer(inp, 64, (7, 7), (2, 2), (3, 3)) # image size goes from 256*256 to 128*128, 64 channels
            relu1 = relu_layer(batch_norm_layer(conv1, training)) # apply batch norm then relu
            res1 = residual_hg(relu1, 128, training) # produce 128 channels from 64 channels
            # max_pool1 = max_pool_layer(res1, (2, 2), (2, 2)) # 2x2 pooling with stride 2x2 => Down-sampling: we half the image height and width to become 64*64
            max_pool1 = linear_layer(res1, 128, training) # apply linear layer
            
            res2 = residual_hg(max_pool1, 128, training)
            res3 = residual_hg(res2, 128, training)
            res4 = residual_hg(res3, 256, training)
        
        inter = res4 # intermediate input of the intermediate hourglass
        last_lin = max_pool1
        nb_channels_last_lin = last_lin.shape[1]
        outputs = [] # the outputs of the hourglasses
        
        for i in range(self.nb_stacks):
            with tf.name_scope("hourglass_{}".format(i)):
                hg_out = self.hourglass(inter, self.n, self.nb_channels, training) # push intermediate input through hourglass
            
            # linear layers
            with tf.name_scope("hourglass_{}_lin_layer_1".format(i)):
                lin1 = linear_layer(hg_out, self.nb_channels, training) # 1st linear layer, same nb output_channels as its input
            
            lin2_out_channels = 256 # nb of channels outputted by 2nd linear layer
            if i == self.nb_stacks - 1: # if this is the last hourglass, then lin2 outputs nb_channels=512
                lin2_out_channels = self.nb_channels
            with tf.name_scope("hourglass_{}_lin_layer_2".format(i)):
                lin2 = linear_layer(lin1, lin2_out_channels, training) # 2nd linear layer, we output 256 channels, except for the last hourglass where we output nb_channels
            
            # produce 3D heatmaps
            with tf.name_scope("hourglass_{}_heatmaps".format(i)):
                heatmaps = conv_layer(lin2, self.nb_joints*self.z_res[i], (1, 1), (1, 1), (0, 0)) # for each joint, produce a 3D heatmap of nb_channels = z_res[i]            
            outputs.append(heatmaps) # store the predicted heatmaps
            
            if(i < self.nb_stacks-1): # if this isn't the last hourglass
                with tf.name_scope("hourglass_{}_transformed_heatmaps".format(i)):
                    transformed = conv_layer(heatmaps, lin2_out_channels+nb_channels_last_lin, (1, 1), (1, 1), (0, 0)) # transform to prepare for add-op next
                with tf.name_scope("hourglass_{}_concat_conv".format(i)):
                    concat = tf.concat([lin2, last_lin], axis=1) # concat the lin2 with last_lin along the channels axis (accross the batch)
                    conv_concat = conv_layer(concat, lin2_out_channels+nb_channels_last_lin, (1, 1), (1, 1), (0, 0))
                with tf.name_scope("hourglass_{}_input".format(i+1)):
                    inter = conv_concat + transformed # input to the next hourglass
            
                # update last_lin for the next iteration
                last_lin = lin2
                nb_channels_last_lin = last_lin.shape[1]
        
        self.outputs = outputs
        
        final_output = outputs[-1]
        self.final_output = final_output
        
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#            outputs_vals = sess.run(outputs)
#            for output in outputs_vals:
#                print(output.shape)
#                
#        return None, None

        # transform the final output to p3d
        with tf.name_scope("heatmaps_to_poses"):
            poses = []
            for all_joints_heatmaps in tf.unstack(final_output): # for each set of 17 heatmaps (1 per joint) of the batch
                joints = []
                z_final_out = self.z_res[-1] # nb of planes in z-axis in the 3D heatmap of each joint
                for i in range(0, self.nb_joints): # for each joint heatmap
                    joint_heatmap = all_joints_heatmaps[i*z_final_out : (i+1)*z_final_out] # extract the 3D heatmap of the joint (i,e the "z_final_out"=64 planes of the joint)
                    joint = tf.reverse(tf.cast(tf.unravel_index(tf.argmax(tf.reshape(joint_heatmap, [-1]), output_type=tf.int32), tf.shape(joint_heatmap)),tf.float32),
                                       axis=[0]) # for each heatmap array "a", flatten array then get argmax index in flattened array then convert flat_index back to 2d index using 
                                                 # unravel_index then cast 2d index to float32, then swap the result (using tf.reverse) since the result is (z, y, x), but we want (x, y, z)
                    joints.append(joint)
                
                poses.append(tf.stack(joints))
                
            p3d_pred = tf.stack(poses)
        
        # in each axis, say z-axis, the range goes from [0, output_shape[0]] pixels and correponds to the metric range [0, 2000] mm so we need to divide the z-axis predictions by output_shape[0]
        # and multiply by 2000 to get the nb of millimeter, then center on the root joint. Likewise for y and x axis
        
        with tf.name_scope("scaling"):
            self.scale_xyz = scale_xyz = tf.constant([self.output_shape[2], self.output_shape[1], self.output_shape[0]], dtype=tf.float32)
            self.metric_shift_xyz = tf.constant([self.shift_x, self.shift_y, self.shift_z], dtype=tf.float32)
            self.metric_scale_xyz = metric_scale_xyz = tf.constant([self.range_x[1]-self.range_x[0], self.range_y[1]-self.range_y[0], self.range_z[1]-self.range_z[0]], dtype=tf.float32)

        
        with tf.name_scope("poses_to_mm_centered"):
            p3d_pred_mm = (p3d_pred/scale_xyz)*metric_scale_xyz
            poses = []
            for p3d in tf.unstack(p3d_pred_mm): # for each predicted pose
                p3d = p3d - p3d[0] # center around root joint
                poses.append(p3d)
            
            p3d_pred_mm_centered = tf.stack(poses)
        
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#            p3d_pred_vals, p3d_pred_mm_vals, p3d_pred_mm_centered_vals = sess.run([p3d_pred, p3d_pred_mm, p3d_pred_mm_centered])
#            print(p3d_pred_vals[0])
#            print(p3d_pred_vals.shape)
#            print(p3d_pred_mm_vals[0])
#            print(p3d_pred_mm_vals.shape)
#            print(p3d_pred_mm_centered_vals[0])
#            print(p3d_pred_mm_centered_vals.shape)
#            
#        return None, None

        print("\n\nBuilt the model in {} s\n\n".format(time.time()-a))

        return outputs, p3d_pred_mm_centered # return all heatmaps of all hourglasses and the centered p3d predictions
        
        
    def hourglass(self, inp, n, nb_channels, training): # recursive definition of an hourglass, n=number of residual_blocks (cubes in paper) till the center of the hourglass
        up_branch = residual_hg(residual_hg(residual_hg(inp, 256, training), 256, training), nb_channels, training) # residual block of 3 residual modules applied on the input
        
        low_branch = max_pool_layer(inp, (2, 2), (2, 2)) # 2x2 pooling of input with stride 2x2 => Down-sampling: we half the input's height and width
        low_branch = residual_hg(residual_hg(residual_hg(low_branch, 256, training), 256, training), 256, training) # residual block of 3 residual modules applied on the max_pool output in low_branch
        
        if n > 1: # while n > 1, we keep performing a recursive call
            low_branch = self.hourglass(low_branch, n-1, nb_channels, training) # recursive call
        else: # n == 1
            low_branch = residual_hg(low_branch, nb_channels, training) # residual block of 1 residual module applied on the low_branch
        
        # starting here, we start building the right half of the hourglass
        low_branch = residual_hg(low_branch, nb_channels, training) # residual block 1 residual module applied on the low_branch
        low_branch = nearest_neighbor_up_sampling_layer(low_branch, (2, 2)) # Up-sample: double the image width and height
        
        return up_branch + low_branch # return sum of up_branch and low_branch

    def compute_loss(self, p3d_gt, all_heatmaps_pred):
        with tf.name_scope("gt_poses_to_heatmaps"):
            p3d_gt_shifted_scaled = ((p3d_gt+self.metric_shift_xyz)/self.metric_scale_xyz)*self.scale_xyz # perform shifting and rescaling to rescaling to final joint heatmap output_shape
            
            outputs_gt = []
            for res in self.z_res: # for each output res along z-axis (i,e 1 per hourglass)
                all_heatmaps = [] # to store all heatmaps across the batch
                for p3d in tf.unstack(p3d_gt_shifted_scaled): # for each pose in the batch
                    heatmaps = [] # heatmaps of 1 pose
                    z_out, x_out, y_out = self.output_shape
                    for joint in tf.unstack(p3d): # for each joint in the pose
                        heatmap = self.gaussian_3d(joint, # the joint is the center of the gaussian
                                                   self.var, # var along x-axis
                                                   self.var, # var along y-axis
                                                   self.var*res/z_out, # var along z-axis (we squeeze the variance using res)
                                                   y_out, x_out, res) # generate the 3d gaussian of shape output_shape
                        heatmaps.append(heatmap)
                    
                    all_heatmaps.append(tf.concat(heatmaps, axis=0))
                
                outputs_gt.append(tf.stack(all_heatmaps))
            
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#            outputs_gt_vals, images = sess.run([outputs_gt, self.inp])
#            image = ((images[0]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
#            image = Image.fromarray(image, "RGB")
#            if not os.path.exists("test_gt_heatmaps"):
#                os.makedirs("test_gt_heatmaps")
#            image.save("test_gt_heatmaps/img.png")
#            for i, out in enumerate(outputs_gt_vals[-1][0]):
#                joint_nb = i//self.z_res[-1]
#                plane_nb = i%self.z_res[-1]
#                out = (out*255).astype("uint8")
#                image = Image.fromarray(out)
#                image = image.resize((256,256))
#                image.save("test_gt_heatmaps/joint_{}_{}.png".format(joint_nb, plane_nb))
#        return None

        with tf.name_scope("loss"):
            total_loss = 0
            for i,heatmaps_pred in enumerate(all_heatmaps_pred):
                loss = tf.losses.mean_squared_error(outputs_gt[i], heatmaps_pred)
                total_loss += loss
            self.loss = total_loss
        
        return total_loss
        
    def gaussian_3d(self, mean, var_x, var_y, var_z, H, W, D):
        with tf.name_scope("gaussian"):
            mu = tf.cast(tf.cast(mean, tf.int32), tf.float32) # remove decimal part due to discretization of the 3D space
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


            
    def get_train_op(self, loss, learning_rate=0.001):
        with tf.name_scope("train"):
            self.global_step = tf.Variable(0, name='global_step',trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, self.global_step)
            return self.train_op, self.global_step
        

        
    
