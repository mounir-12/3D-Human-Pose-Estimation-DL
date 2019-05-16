import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def conv_layer(inp, in_channels, out_channels, filter_H=1, filter_W=1, stride_H=1, stride_W=1, pad_H=0, pad_W=0, use_biases=False): 
    """ input expected of shape [batch_size, in_height, in_width, in_channels] i,e channels last"""
    
    filter_shape = [filter_H, filter_W, in_channels, out_channels] # the filter shape
        
    weights = tf.Variable(shape=filter_shape, initializer=xavier_initializer()) # create filter variable
    
    strides = [1, stride_H, stride_W, 1] # strides shape
    
    if pad_H > 0 or pad_W > 0: # need to pad
        paddings = [[0,0], [pad_H, pad_H], [pad_W, pad_W], [0,0]] # pad only along the height and width dimension, no padding along the batch and channels dimensions
        inp = tf.pad(inp, paddings)
        
    layer_out = tf.nn.conv2d(input=input_layer, filter=weights, strides=strides, padding=pad)
    
    if use_biases: # add biases
        biases_shape=[num_output_channels]
        biases = tf.Variable(shape=biases_shape, initializer=xavier_initializer())
        layer_out += biases

    return layer_out

def relu_layer(inp):
    return tf.nn.relu(inp)

def batch_norm_layer(inp, momentum=0.99, epsilon=0.001, training=False):
    return tf.layers.batch_normalization(inputs=inp, momentum=momentum, epsilon=epsilon, training=training)
