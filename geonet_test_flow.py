from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
import cv2
from geonet_model import *
from data_loader import DataLoader
import sys
sys.path.insert(0, './kitti_eval/flow_tool/')
import flowlib as fl

def test_flow(opt):

    ##### load testing list #####
    with open(opt.dataset_dir + "test_flow.txt", 'r') as f:
        test_files = f.readlines()
        input_list = []
        for seq in test_files:
            seq = seq.split(' ')
            input_list.append(opt.dataset_dir+seq[0]+'/'+seq[1][:-1])

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    # TODO: currently assuming batch_size = 1
    assert opt.batch_size == 1
    
    tgt_image_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                        opt.img_height, opt.img_width, 3],
                        name='tgt_input')
    src_image_stack_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                        opt.img_height, opt.img_width, opt.num_source * 3],
                        name='src_stack_input')
    intrinsics = tf.placeholder(tf.float32, [opt.batch_size, 3, 3],
                        name='intrinsics_input')
    loader = DataLoader(opt)
    intrinsics_ms = loader.get_multi_scale_intrinsics(intrinsics, opt.num_scales)

    # currently assume a sequence is fed and the tgt->src_id flow is computed
    src_id = int(opt.num_source // 2)
    bs = opt.batch_size
    model = GeoNetModel(opt, tgt_image_uint8, src_image_stack_uint8, intrinsics_ms)
    fetches = {}
    fetches["pred_flow"] = model.fwd_full_flow_pyramid[0][bs*src_id:bs*(src_id+1)]

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go! #####
    output_file = opt.output_dir + '/' + os.path.basename(opt.init_ckpt_file)
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    binary_dir = os.path.join(output_file, 'binary')
    color_dir  = os.path.join(output_file, 'color')
    png_dir    = os.path.join(output_file, 'png')
    if (not os.path.exists(binary_dir)):
        os.makedirs(binary_dir)
    if (not os.path.exists(color_dir)):
        os.makedirs(color_dir)
    if (not os.path.exists(png_dir)):
        os.makedirs(png_dir)

    with tf.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        img_num = len(input_list)

        for tgt_idx in range(img_num):
            if (tgt_idx+1) % 100 == 0:
                print('processing: %d/%d' % (tgt_idx+1, img_num))            
            image_seq = cv2.imread(input_list[tgt_idx]+'.jpg')
            tgt_image, src_image_stack = unpack_image_sequence(image_seq, 
                                         opt.img_height, opt.img_width, opt.num_source)
            with open(input_list[tgt_idx]+'_cam.txt', 'r') as cf:
                cam_file = cf.readlines()
            cam_file = cam_file[0].split(',')
            cam_file = np.array([float(d) for d in cam_file])
            cam_file = np.reshape(cam_file, (3,3))

            pred = sess.run(fetches, feed_dict={tgt_image_uint8: tgt_image[None, :, :, :], 
                                                src_image_stack_uint8: src_image_stack[None, :, :, :],
                                                intrinsics: cam_file[None,:,:]})
            pred_flow=pred['pred_flow'][0]

            # save flow
            flow_fn = '%.6d.png' % tgt_idx
            color_fn    = os.path.join(color_dir, flow_fn)
            color_flow  = fl.flow_to_image(pred_flow)
            color_flow  = cv2.cvtColor(color_flow, cv2.COLOR_RGB2BGR)
            color_flow  = cv2.imwrite(color_fn, color_flow)

            png_fn      = os.path.join(png_dir, flow_fn)
            mask_blob   = np.ones((opt.img_height, opt.img_width), dtype = np.uint16)
            fl.write_kitti_png_file(png_fn, pred_flow, mask_blob)

            binary_fn   = flow_fn.replace('.png', '.flo')
            binary_fn   = os.path.join(binary_dir, binary_fn)
            fl.write_flow(pred_flow, binary_fn)

def unpack_image_sequence(image_seq, img_height, img_width, num_source):
    # Assuming the center image is the target frame
    half_seq_width = int(img_width * (num_source//2))
    tgt_image = image_seq[:, half_seq_width:half_seq_width+img_width, :]
    # Source frames before the target frame
    src_image_1 = image_seq[:, :half_seq_width, :]
    # Source frames after the target frame
    src_image_2 = image_seq[:, half_seq_width+img_width:, :]
    src_image_seq = np.hstack((src_image_1, src_image_2))
    # Stack source frames along the color channels (i.e. [H, W, N*3]
    src_image_stack = np.dstack([src_image_seq[:,i*img_width:(i+1)*img_width,:] \
                      for i in range(num_source)])
    return tgt_image, src_image_stack