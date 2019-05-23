import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

""" all functions below expect input in channels first format for perfomance on GPU """

def padding_layer(inp, padding):
    with tf.name_scope("padding_layer"):
        pad_H = padding[0]
        pad_W = padding[1]
        if pad_H > 0 or pad_W > 0: # perform padding only if padding by 1 pixel or more
            paddings = [[0, 0], [0, 0], [pad_H, pad_H], [pad_W, pad_W]] # pad only along the height and width dimension, no padding along the batch and channels dimensions
            return tf.pad(inp, paddings)
        else:
            return inp

def conv_layer(inp, out_channels, filter_size=1, strides=1, padding="same", use_bias=False): 
    with tf.name_scope("conv_layer"):
        if isinstance(padding, (list, tuple)): # if a list/tuple passed
            inp = padding_layer(inp, padding)
            padding = "valid"

        return tf.layers.conv2d(inputs=inp, filters=out_channels, kernel_size=filter_size, strides=strides, padding=padding, 
                                data_format="channels_first", use_bias=use_bias, kernel_initializer=xavier_initializer(), 
                                bias_initializer=xavier_initializer())

def relu_layer(inp):
    with tf.name_scope("relu_layer"):
        return tf.nn.relu(inp)

def batch_norm_layer(inp, training, momentum=0.99, epsilon=0.001):
    with tf.name_scope("batch_norm_layer"):
        return tf.layers.batch_normalization(inputs=inp, axis=1, momentum=momentum, epsilon=epsilon, training=training) # axis=1 because channels first
    
def max_pool_layer(inp, pool_size=1, strides=1, padding="valid"):
    with tf.name_scope("max_pool_layer"):
        if isinstance(padding, (list, tuple)): # if a list/tuple passed
            inp = padding_layer(inp, padding)
            padding = "valid"
            
        return tf.layers.max_pooling2d(inp, pool_size=pool_size, strides=strides, padding=padding, data_format="channels_first")
    
def nearest_neighbor_up_sampling_layer(inp, factor):
    with tf.name_scope("nearest_neighbor_up_sampling_layer"):
#        return tf.keras.layers.UpSampling2D(size=factor, data_format="channels_first", interpolation='nearest').apply(inp)
        inp = tf.transpose(inp, [0, 2, 3, 1]) # convert to channels_last format
        
        new_H = tf.shape(inp)[1] * factor[0] # the new heigh
        new_W = tf.shape(inp)[2] * factor[1] # the new width
        out =  tf.image.resize_nearest_neighbor(inp, (new_H, new_W)) # resize images with nearest neighbor up sampling

        return tf.transpose(out, [0, 3, 1, 2]) # convert to channels_first format and return result
    
# --------------------------------------- Stacked Hourglass Model functions --------------------------------------------------------
    
def conv_block_hg(inp, out_channels, training): # Main convolutional block of hourglass model
    with tf.name_scope("conv_block_hg"):
        tmp = batch_norm_layer(inp, training)
        tmp = relu_layer(tmp)
        tmp = conv_layer(tmp, out_channels/2, (1, 1)) # 1x1 kernel, 1x1 stride so the input width and height are conserved and "out_channels/2" channels are produced
        
        tmp = batch_norm_layer(tmp, training)
        tmp = relu_layer(tmp)
        tmp = conv_layer(tmp, out_channels/2, (3, 3), (1, 1), (1, 1)) # 3x3 kernel 1x1 stride, 1x1 padding, so the input width and height are conserved and "out_channels/2" channels are produced
        
        
        tmp = batch_norm_layer(tmp, training)
        tmp = relu_layer(tmp)
        out = conv_layer(tmp, out_channels, (1, 1)) # 1x1 kernel, 1x1 stride so the input width and height are conserved and "out_channels" channels are produced
        
        return out

def skip_layer_hg(inp, out_channels): # Skip Layer of hourglass
    with tf.name_scope("skip_layer_hg"):
        in_channels = inp.get_shape()[1].value # get nb of channels of input
        if in_channels == out_channels: # if same number of channels
            return inp # just the input
        else: # otherwise
            return conv_layer(inp, out_channels, (1, 1)) # 1x1 kernel, 1x1 stride, "out_channels" channels are produced
            
def residual_hg(inp, out_channels, training): # residual module of hourglass
    with tf.name_scope("residual_module_hg"):
        out_1 = conv_block_hg(inp, out_channels, training) # push the input through the conv block
        out_2 = skip_layer_hg(inp, out_channels) # push the input through the skip layer
        return out_1 + out_2 # sum up and return result

def residual_block_hg(inp, out_channels, nb_times, training): # residual block consisting of residual_modules stacked nb_times
    with tf.name_scope("residual_block_hg"):
        out = inp
        for _ in range(max(1, nb_times)): # apply residual module at least 1 time
            out = residual_hg(out, out_channels, training)
        return out
        
def linear_layer(inp, out_channels, training):
    with tf.name_scope("linear_layer"):
        inp = conv_layer(inp, out_channels, (1, 1), (1, 1), (0, 0)) # 1x1 kernel, 1x1 stride, 0x0 padding so input width and height are conserved and "out_channels" channels are produced
        return relu_layer(batch_norm_layer(inp, training)) # perform batch norm then relu



 
