import tensorflow as tf
from tf.keras.layers import Conv2D, BatchNormalization, ReLU, Add
from tf.keras.models import Sequential
from tf.contrib.layers import xavier_initializer

class StackedHourglass:
    def __init__(nb_channels, nb_modules, hourglass_half_size, nb_stacks):
        self.nb_modules = nb_modules
        self.n = hourglass_half_size
        self.nb_stacks = nb_stacks
        self.nb_channels = nb_channels
        
    def build_model(input_shape, ):
        self.input 
    
    def apply_hourglass(inp, n, channels):
        
    def apply_residual():
        
    def conv_block(inp, out_channels): # Main convolutional block
        return Sequential()
                .add(BatchNormalization())
                .add(ReLU())
                .add(Conv2D(filters=out_channels/2, 
                            kernel_size=(1, 1), 
                            strides=(1, 1), 
                            use_bias=False, 
                            kernel_initializer=xavier_initializer())) # 1x1 kernel, 1x1 stride so the input width and height are conserved and "out_channels/2" channels are produced
                .add(BatchNormalization())
                .add(ReLU())
                .add(Conv2D(filters=out_channels/2, 
                            kernel_size=(3, 3), 
                            strides=(1, 1),
                            padding="same" 
                            use_bias=False, 
                            kernel_initializer=xavier_initializer())) # 3x3 kernel 1x1 stride, padding="same", so the input width and height are conserved and "out_channels/2" channels are produced
                .add(BatchNormalization())
                .add(ReLU())
                .add(Conv2D(filters=out_channels, 
                            kernel_size=(1, 1), 
                            strides=(1, 1), 
                            use_bias=False, 
                            kernel_initializer=xavier_initializer())) # 1x1 kernel, 1x1 stride so the input width and height are conserved and "out_channels" channels are produced
                .apply(inp) # apply the block on the input

    def skip_layer(inp, out_channels): # Skip Layer
        in_channels = input_layer.get_shape()[-1].value
        if in_channels == out_channels: # if same number of channels
            return tf.identity(inp) # just return a copy of the input
        else # otherwise
            return Conv2D(filters=out_channels, 
                          kernel_size=(1, 1), 
                          strides=(1, 1), 
                          use_bias=False, 
                          kernel_initializer=xavier_initializer()) # 1x1 kernel, 1x1 stride, "out_channels" channels are produced
                    .apply(inp) # perform a conv layer to produce "out_channels" channels from the input
                
    def residual(inp, out_channels):
        out_1 = conv_block(inp, out_channels) # push the input through the conv block
        out_2 = skip_layer(inp, out_channels) # push the input through the skip layer
        return Add([out_1, out_2]) # sum up the 2 results
        
    
