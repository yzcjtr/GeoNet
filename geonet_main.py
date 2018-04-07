from __future__ import division
import os
import time
import random
import pprint
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from geonet_model import *
from geonet_test_depth import *
from geonet_test_pose import *
from geonet_test_flow import *
from data_loader import DataLoader

flags = tf.app.flags
flags.DEFINE_string("mode",                         "",    "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "",    "Dataset directory")
flags.DEFINE_string("init_ckpt_file",             None,    "Specific checkpoint file to initialize from")
flags.DEFINE_integer("batch_size",                   4,    "The size of of a sample batch")
flags.DEFINE_integer("num_threads",                 32,    "Number of threads for data loading")
flags.DEFINE_integer("img_height",                 128,    "Image height")
flags.DEFINE_integer("img_width",                  416,    "Image width")
flags.DEFINE_integer("seq_length",                   3,    "Sequence length for each example")

##### Training Configurations #####
flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate",             0.0002,    "Learning rate for adam")
flags.DEFINE_integer("max_to_keep",                 20,    "Maximum number of checkpoints to save")
flags.DEFINE_integer("max_steps",               300000,    "Maximum number of training iterations")
flags.DEFINE_integer("save_ckpt_freq",            5000,    "Save the checkpoint model every save_ckpt_freq iterations")
flags.DEFINE_boolean("retrain",                   True,    "Retrain from step zero when init checkpoint given")
flags.DEFINE_float("alpha_recon_image",           0.85,    "Alpha weight between SSIM and L1 in reconstruction loss")

##### Configurations about DepthNet & PoseNet of GeoNet #####
flags.DEFINE_string("dispnet_encoder",      "resnet50",    "Type of encoder for dispnet, vgg or resnet50")
flags.DEFINE_boolean("scale_normalize",          False,    "Spatially normalize depth prediction")
flags.DEFINE_float("rigid_warp_weight",            1.0,    "Weight for warping by rigid flow")
flags.DEFINE_float("disp_smooth_weight",           0.5,    "Weight for disp smoothness")

##### Configurations about ResFlowNet of GeoNet (or DirFlowNetS) #####
flags.DEFINE_string("flownet_type",         "residual",    "type of flownet, residual or direct")
flags.DEFINE_float("flow_warp_weight",             1.0,    "Weight for warping by full flow")
flags.DEFINE_float("flow_smooth_weight",           0.2,    "Weight for flow smoothness")
flags.DEFINE_float("flow_consistency_weight",      0.2,    "Weight for bidirectional flow consistency")
flags.DEFINE_float("flow_consistency_alpha",       3.0,    "Alpha for flow consistency check")
flags.DEFINE_float("flow_consistency_beta",       0.05,    "Beta for flow consistency check")

##### Testing Configurations #####
flags.DEFINE_string("output_dir",                 None,    "Test result output directory")
flags.DEFINE_string("depth_test_split",        "eigen",    "KITTI depth split, eigen or stereo")
flags.DEFINE_integer("pose_test_seq",                9,    "KITTI Odometry Sequence ID to test")


opt = flags.FLAGS

def train():

    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    # Data Loader
    loader = DataLoader(opt)
    tgt_image, src_image_stack, intrinsics = loader.load_train_batch()

    # Build Model
    model = GeoNetModel(opt, tgt_image, src_image_stack, intrinsics)

    # Train Op
    if opt.mode == 'train_flow' and opt.flownet_type == "residual":
        # we pretrain DepthNet & PoseNet, then finetune ResFlowNetS
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "flow_net")
        vars_to_restore = slim.get_variables_to_restore(include=["depth_net", "pose_net"])
    else:
        train_vars = [var for var in tf.trainable_variables()]
        vars_to_restore = slim.get_model_variables()
            
    optim = tf.train.AdamOptimizer(opt.learning_rate, 0.9)
    train_op = slim.learning.create_train_op(model.total_loss, optim,
                                             variables_to_train=train_vars)

    # Parameter Count
    parameter_count = 0
    print("Trainable variables: ")
    for var in train_vars:
        print(var.name)
        parameter_count += np.array(var.get_shape().as_list()).prod()
    print("Parameter count: {}".format(parameter_count))

    # Global Step
    global_step = tf.Variable(1, name='global_step', trainable=False)

    # Saver
    train_saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                 [global_step], max_to_keep=opt.max_to_keep)

    if opt.init_ckpt_file != None:
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                                         opt.init_ckpt_file, vars_to_restore)

    # Session
    sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, saver=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with sv.managed_session(config=config) as sess:

        if opt.init_ckpt_file != None:
            sess.run(init_assign_op, init_feed_dict)

        # if opt.retrain:
        #     sess.run(global_step.assign(1))

        # Training Loop
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, opt.max_steps):        
            _, loss_value = sess.run([train_op, model.total_loss])
            if step % 100 == 0:
                time_per_iter = (time.time() - start_time) / 100
                start_time = time.time()
                print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f' \
                      % (step, time_per_iter, loss_value))
            if step % opt.save_ckpt_freq == 0:
                train_saver.save(sess, opt.checkpoint_dir + 'model', global_step=step)

        train_saver.save(sess, opt.checkpoint_dir + 'model', global_step=opt.max_steps)

def main(_):

    opt.num_source = opt.seq_length - 1
    opt.num_scales = 4

    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'test_depth']
    opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'test_pose']

    if opt.mode in ['train_rigid', 'train_flow']:
        train()
    elif opt.mode == 'test_depth':
        test_depth(opt)
    elif opt.mode == 'test_pose':
        test_pose(opt)
    elif opt.mode == 'test_flow':
        test_flow(opt)

if __name__ == '__main__':
    tf.app.run()
