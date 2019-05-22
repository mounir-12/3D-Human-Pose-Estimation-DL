import tensorflow as tf
import tensorflow_hub as hub


class Resnet_50:
	def __init__(self, nb_joints=17, tfhub_module, loss_type=1):
		self.nb_joints = nb_joints
		self.tfhub_module = tfhub_module
		self.loss_type = loss_type

	def __call__(self, inp):
		module = hub.load_module_spec(tfhub_module, trainable=True, tags={"train"})
		height,width = hub.get_expected_image_size(module)
		output_size = 3*self.nb_joints
		m = hub.Module(module)
		self.feature_tensor = m(inputs = dict(images=inp, batch_norm_momentum=0.99), signature="image_feature_vector_with_bn_hparams")
		, feature_tensor_size = feature_tensor.get_shape().as_list()
		initial_value = tf.truncated_normal([feature_tensor_size, output_size], stddev=0.001)
		self.weights = tf.Variable(initial_value, name='final_weights')
		self.biases = tf.Variable(tf.zeros([output_size]), name='final_biases')

		pose = tf.matmul(feature_tensor, self.weights) + self.biases

		self.pose =  tf.nn.relu(pose)

		return self.pose

	def compute_loss(self, pose_gt, pose_pred):

		if(self.loss_type == 1):
			return(tf.tf.losses.absolute_difference(pose_gt, pose_pred))



	def train_op(self, loss, learning_rate=0.001):
		self.global_step = tf.Variable(0, name='global_step',trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, self.global_step)
        return self.train_op, self.global_step