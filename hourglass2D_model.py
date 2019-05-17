import tensorflow as tf
from layers import conv_layer, relu_layer, max_pool_layer, nearest_neighbor_up_sampling_layer, residual_hg, residual_block_hg, linear_layer

class StackedHourglass:
    def __init__(self, nb_joins=17, nb_channels=256, nb_modules=1, hourglass_half_size=4, nb_stacks=1):
        self.dtype = tf.float32 # data type to use
        
        self.images_shape = (3, 256, 256) # the shape of input image with channels first (here: 3*256*256)
        self.nb_joins = nb_joins # nb of joins to be predicted
        self.nb_channels = nb_channels # nb of channels in the hourglass (expected as input to hourglass and preserved inside it) 
        self.nb_modules = nb_modules # nb of residual modules stacked in 1 residual block (which is the basic block in the network)
        self.n = hourglass_half_size # nb of residual blocks applied till the center of the hourglass
        self.nb_stacks = nb_stacks # nb of hourglasses stacked in the network
        
        self.batch_shape = list(self.images_shape).copy() # copy shape
        self.batch_shape.insert(0, None) # prepend None for variable batch size
        
    def __call__(self, inp, training=False):
#        self.inp = tf.placeholder(self.dtype, shape=self.batch_shape) # the input batch of images
#        self.training = tf.placeholder(tf.bool) # boolean used in batch_normalization layer, is True when training and False when predicting

        inp = conv_layer(inp, 64, (7, 7), (2, 2), (3, 3))
        inp = relu_layer(inp)
        inp = max_pool_layer(inp, (2, 2), (2, 2)) # 2x2 pooling with stride 2x2 => Down-sampling: we half the image height and width
        
        inp = residual_hg(inp, 128, training)
        inp = residual_hg(inp, self.nb_channels, training)
        
        inter = inp # intermediate input of the intermediate hourglass
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
        return outputs[-1] # return the last outputted heatmap
        
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
        
    def compute_loss(self, labels):
        return None
            
        

        

        
    
