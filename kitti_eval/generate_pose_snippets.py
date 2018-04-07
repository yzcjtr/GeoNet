from __future__ import division
import os
import math
import scipy.misc
import numpy as np
import argparse
from glob import glob
from pose_evaluation_utils import mat2euler, dump_pose_seq_TUM

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="path to kitti odometry dataset")
parser.add_argument("--output_dir",  type=str, help="path to output pose snippets")
parser.add_argument("--seq_id",      type=int, default=9, help="sequence id to generate groundtruth pose snippets")
parser.add_argument("--seq_length",  type=int, default=5, help="sequence length of pose snippets")
args = parser.parse_args()

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

def main():
    pose_gt_dir = args.dataset_dir + 'poses/'
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    seq_dir = os.path.join(args.dataset_dir, 'sequences', '%.2d' % args.seq_id)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (args.seq_id, n) for n in range(N)]
    with open(args.dataset_dir + 'sequences/%.2d/times.txt' % args.seq_id, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    with open(pose_gt_dir + '%.2d.txt' % args.seq_id, 'r') as f:
        poses = f.readlines()
    poses_gt = []
    for pose in poses:
        pose = np.array([float(s) for s in pose[:-1].split(' ')]).reshape((3,4))
        rot = np.linalg.inv(pose[:,:3])
        tran = -np.dot(rot, pose[:,3].transpose())
        rz, ry, rx = mat2euler(rot)
        poses_gt.append(tran.tolist() + [rx, ry, rz])
    poses_gt = np.array(poses_gt)

    max_src_offset = (args.seq_length - 1)//2
    for tgt_idx in range(N):
        if not is_valid_sample(test_frames, tgt_idx, args.seq_length):
            continue
        if tgt_idx % 100 == 0:
            print('Progress: %d/%d' % (tgt_idx, N))
        pred_poses = poses_gt[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        out_file = args.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
        dump_pose_seq_TUM(out_file, pred_poses, curr_times)

main()

