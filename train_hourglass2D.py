"""Copyright (c) 2019 AIT Lab, ETH Zurich, Xu Chen

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import tensorflow as tf
import numpy as np
import os, sys, gc, glob
from tqdm import trange
import utils
from data import create_dataloader_train, create_dataloader_test
from hourglass2D_model import StackedHourglass
import time
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--num_epochs', type = int, default = 2, help = 'number of training epochs')
parser.add_argument('--continue_training', help = 'whether to continue training from the last checkpoint of the last experiment', action="store_true") # by default, we don't continue training
parser.add_argument('--batch_size', type = int, default = 4, help = 'size of training batch')
parser.add_argument('--learning_rate', type = float, default = 2.5*10**(-4), help = 'learning rate for the optimizer')
parser.add_argument('--nb_stacks', type = float, default = 8, help = 'number of stacked hourglasses')
parser.add_argument('--decay_lr', help='whether to deacy learning rate during training', action="store_true") # by default, we don't decay the learning rate

parser.add_argument('--log_iter_freq', type = int, default = 100, help = 'number of iterations between training logs')
parser.add_argument('--valid_iter_freq', type = int, default = 500, help = 'number of iterations between validation steps')
parser.add_argument('--valid_steps', type = int, default = 10, help = 'number of validation batches per validation step')
parser.add_argument('--save_iter_freq', type = int, default = 2000, help = 'number of iterations between saving model checkpoints')
parser.add_argument('--valid_samples', type = int, default = 2188, help = 'number of samples in the validation set')

parser.add_argument('--no_test_every_epoch', help = 'whether to evaluate test step after each train epoch', action="store_true") # by default, we test every epoch
parser.add_argument('--no_data_aug', help = 'whether to augment data using image rotations', action="store_true") # by default, we augment data
parser.add_argument('--batches_to_prefetch', type = int, default = 20, help = 'number of batches to prefetch')


args = parser.parse_args()

NUM_SAMPLES= 312188
NUM_SAMPLES_TEST = 10987
CONTINUE_TRAINING = args.continue_training

# Train parameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DECAY_LR = args.decay_lr
LOG_ITER_FREQ = args.log_iter_freq
VALID_ITER_FREQ = args.valid_iter_freq
VALID_STEPS = args.valid_steps # number of validation batches to use to compute the mean validation loss and mpjpe
SAVE_ITER_FREQ = args.save_iter_freq
VALID_SAMPLES = args.valid_samples
TEST_EVERY_EPOCH = not args.no_test_every_epoch

# Data parameters
SHUFFLE = True
DATA_TO_LOAD = "pose2d"
AUGMENT_DATA = not args.no_data_aug
BATCHES_TO_PREFETCH = args.batches_to_prefetch

# Model parameters
NB_STACKS=args.nb_stacks
SIGMA=1

# Paths
CURR_DIR = "."
if CONTINUE_TRAINING: # continue training from last training loop
    list_of_files = glob.glob(os.path.join(CURR_DIR, "log_HG2D", "*"))
    LOG_PATH = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment will be our log path
else:
    LOG_PATH = os.path.join(CURR_DIR, "log_HG2D", utils.timestamp())

CHECKPOINTS_PATH = os.path.join(LOG_PATH, "checkpoints")
CLUSTER_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"
if os.path.exists(CLUSTER_PATH):
    DATA_PATH = CLUSTER_PATH
else:
    DATA_PATH = CURR_DIR

# printing all above parameters
print("\n")
print("Run infos:")
print("    NUM_EPOCHS: {}".format(NUM_EPOCHS))
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    BATCHES_TO_PREFETCH: {}".format(BATCHES_TO_PREFETCH))
print("    AUGMENT_DATA: {}".format(AUGMENT_DATA))
print("    LEARNING_RATE: {}".format(LEARNING_RATE))
print("    DECAY_LR: {}".format(DECAY_LR))
print("    LOG_DIR: {}".format(LOG_PATH))
print("    CONTINUE_TRAINING: {}".format(CONTINUE_TRAINING))
print("    TEST_EVERY_EPOCH: {}".format(TEST_EVERY_EPOCH))
print("    NB_STACKS: {}".format(NB_STACKS))
print("\n")
sys.stdout.flush()

#sys.exit(0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:
    
    # load dataset of batched pairs (image, pose), means and stddev
    train_ds, valid_ds, VALID_SIZE, _, _ = create_dataloader_train(data_root=DATA_PATH, batch_size=BATCH_SIZE, valid_size=VALID_SAMPLES,
                                                                   batches_to_prefetch=BATCHES_TO_PREFETCH, data_to_load=DATA_TO_LOAD, shuffle=SHUFFLE, augment=AUGMENT_DATA)
    NUM_SAMPLES = NUM_SAMPLES - VALID_SIZE # update NUM_SAMPLES
    im_train, p2d_gt_train = train_ds
    im_valid, p2d_gt_valid = valid_ds
    
#    for j in range(5):                                                       
#        images, p2d_gt_vals = sess.run([im_train, p2d_gt_train])
#        
#        for index in range(BATCH_SIZE):
#            image = ((images[index]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
#            image = np.asarray(Image.fromarray(image, "RGB")) # necessary conversion for cv2
#            save_dir = os.path.join(LOG_PATH, "augmented_samples")
#            utils.save_p2d_image(image, p2d_gt_vals[index], None, save_dir, j*BATCH_SIZE+index) # save the a random image of the batch
#    
#    sys.exit(0)
    
    print("Building model, NB_STACKS: {}, Sigma: {} ...\n".format(NB_STACKS, SIGMA))
    sys.stdout.flush()

    # define and build the model
    with tf.variable_scope("model", reuse=False):
        model_train = StackedHourglass(nb_stacks=NB_STACKS, sigma=SIGMA)
        all_heatmaps_pred_train, p2d_pred_train = model_train(im_train, True)

#    sys.exit(0)

    with tf.variable_scope("model", reuse=True):
        model_valid = StackedHourglass(nb_stacks=NB_STACKS, sigma=SIGMA)
        all_heatmaps_pred_valid, p2d_pred_valid = model_valid(im_valid, False)
    
    # test data
    im_test = create_dataloader_test(data_root=DATA_PATH) # load test data with batch_size=1
    with tf.variable_scope("model", reuse=True):
        model_test = StackedHourglass(nb_stacks=NB_STACKS, sigma=SIGMA)
        all_heatmaps_pred_test, p2d_pred_test = model_test(im_test, False)
    
#    sys.exit(0)
    
    # compute loss
    print("Loss...")
    sys.stdout.flush()
    loss_train = model_train.compute_loss(p2d_gt_train, all_heatmaps_pred_train)
    loss_valid = model_valid.compute_loss(p2d_gt_valid, all_heatmaps_pred_valid)
#    sys.exit(0)

    # define trainer
    print("Train op...")
    sys.stdout.flush()
    if DECAY_LR: # perform learning rate decay
        decay_steps = NUM_SAMPLES // BATCH_SIZE # decay every end of epoch
        decay_rate = 0.5 # divide learning by 2 every decay_steps
        print("Learning rate decaying every {} steps by a factor of {}".format(decay_steps, 1.0/decay_rate))
        sys.stdout.flush()
        train_op, global_step, learning_rate= model_train.get_train_op(loss_train, decay_steps=decay_steps, decay_rate=decay_rate, learning_rate=LEARNING_RATE)
    else:
        print("Learning rate fixed at {}".format(LEARNING_RATE))
        sys.stdout.flush()
        train_op, global_step, learning_rate= model_train.get_train_op(loss_train, learning_rate=LEARNING_RATE)
#    sys.exit(0)

    print("MPJPE ...")
    sys.stdout.flush()
    mpjpe_train = utils.compute_MPJPE_2D(p2d_pred_train,p2d_gt_train)
    mpjpe_valid = utils.compute_MPJPE_2D(p2d_pred_valid,p2d_gt_valid)

    # visualization related
    
    print("Summaries ...")
    sys.stdout.flush()
    train_summary = tf.summary.merge([tf.summary.scalar("train_loss", loss_train),
                                     tf.summary.scalar("train_mpjpe", mpjpe_train),
                                     tf.summary.scalar("learning_rate", learning_rate)])
    
    valid_loss_pl = tf.placeholder(dtype=tf.float32)
    valid_mpjpe_pl = tf.placeholder(dtype=tf.float32)
    valid_summary = tf.summary.merge([tf.summary.scalar("valid_loss", valid_loss_pl), 
                                      tf.summary.scalar("valid_mpjpe", valid_mpjpe_pl)])
    
    print("Initializing summaries writer ...")
    sys.stdout.flush()
    
    if CONTINUE_TRAINING: # if continuing training, no need to write the graph again to the events file
        writer = tf.summary.FileWriter(CHECKPOINTS_PATH)
    else: # otherwise, write graph
        writer = tf.summary.FileWriter(CHECKPOINTS_PATH, sess.graph)
    
#    sys.exit(0)

    # define model saver
    print("Initializing model saver ...")
    sys.stdout.flush()
    saver = tf.train.Saver(tf.global_variables())
    
    if CONTINUE_TRAINING: # restore variables from saved model
        print("\nRestoring Model ...")
        saver.restore(sess,tf.train.latest_checkpoint(CHECKPOINTS_PATH)) # restore model from last checkpoint
        global_step_val = sess.run(global_step)
        for filename in glob.glob(os.path.join(CHECKPOINTS_PATH, "model*")): # remove all previously saved checkpoints (for Leonhard since limited disk space given)
            os.remove(filename)
        saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val) # save the restored model (i,e keep the last checkpoint in this new run)
        print("Model restored from ", CHECKPOINTS_PATH)
        print("Continuing training for {} epochs ... ".format(NUM_EPOCHS))
        print("Global_step: {}\n".format(global_step_val))
        sys.stdout.flush()
    else: # initialize using initializers
        print("\nInitializing Variables")
        sys.stdout.flush()
        tf.global_variables_initializer().run()
    
#    sys.exit(0)

    # training loop
    print("Training start ...\n")
    sys.stdout.flush()
    # training loop
    with trange(NUM_EPOCHS * (NUM_SAMPLES // BATCH_SIZE)) as t:
        for i in t:

	        # display training status
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES # nb of epochs completed (e,g 1.5 => one epoch and a half)
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES # nb of images processed in current epoch
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) ) # update displayed info, iter_percent = percentage of completion of current iteration (i,e epoch)

            # vis
            if (i+1) % VALID_ITER_FREQ == 0: # validation
                gc.collect() # free-up memory
                valid_loss = valid_mpjpe = 0
                global_step_val = sess.run(global_step) # get the global step value
                for j in range(VALID_STEPS):
                    if j == 0: # when j == 0 we also output an image for visualization
                        images, p2d_gt_vals, p2d_pred_vals, loss_val, mpjpe_val = sess.run([im_valid, p2d_gt_valid, p2d_pred_valid, 
                                                                                           loss_valid, mpjpe_valid]) # get valid loss and mpjpe on 1 batch
                        
                        index = 0 # np.random.randint(BATCH_SIZE) # pick a random index to visualize
                        image = ((images[index]+1)*128.0).transpose(1,2,0).astype("uint8") # unnormalize, put in channels_last format and cast to uint8
                        image = np.asarray(Image.fromarray(image, "RGB")) # necessary conversion for cv2
                        save_dir = os.path.join(LOG_PATH, "valid_samples")
                        utils.save_p2d_image(image, p2d_gt_vals[index], p2d_pred_vals[index], save_dir, global_step_val) # save the a random image of the batch with its predicted pose
                    else:
                        loss_val, mpjpe_val = sess.run([loss_valid, mpjpe_valid]) # get valid loss and mpjpe on 1 batch
                    
                    valid_loss += (loss_val / VALID_STEPS) # add to the mean
                    valid_mpjpe += (mpjpe_val / VALID_STEPS) # add to the mean
                
                summary = sess.run(valid_summary, {valid_loss_pl: valid_loss, valid_mpjpe_pl: valid_mpjpe})
                writer.add_summary(summary, global_step_val)
            
            if (i+1) % LOG_ITER_FREQ == 0: # if it's time log train summaries
                _, summary = sess.run([train_op, train_summary]) # get train summaries
                global_step_val = sess.run(global_step)
                writer.add_summary(summary, global_step_val)
            
            else: # otherwise, just train
                _, = sess.run([train_op])

            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                global_step_val = sess.run(global_step) # get the global step value
                saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val)
                gc.collect() # free-up memory once model saved
                
            if TEST_EVERY_EPOCH and (i+1) % (NUM_SAMPLES // BATCH_SIZE) == 0:
                print("End of epoch, saving model ...")
                sys.stdout.flush()
                global_step_val = sess.run(global_step) # get the global step value
                saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val)
                gc.collect() # free-up memory once model saved
                print("Predicting on test set...")
                sys.stdout.flush()
                predictions = None
                with trange(NUM_SAMPLES_TEST) as t_test: # generate predictions for all images
                    for j in t_test:
                        p2d_out_value = sess.run([p2d_pred_test])

                        if predictions is None:
                            predictions = p2d_out_value
                        else:
                            predictions = np.concatenate([predictions,p2d_out_value],axis=0)
                
                predictions = predictions.reshape([-1, 34])
                print("\nPredictions shape:", predictions.shape)
                sys.stdout.flush()
                            
                predictions_dir = os.path.join(LOG_PATH, "predictions")
                if not os.path.exists(predictions_dir):
                    os.makedirs(predictions_dir)
                
                utils.save_2d_pred(predictions, os.path.join(predictions_dir, "p2d_test_step_{}.csv".format(global_step_val)))
    
    
    global_step_val = sess.run(global_step) # get the global step value
    saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val) # save at the end of training
