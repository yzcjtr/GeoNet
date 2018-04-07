from __future__ import division
import cv2
import os
import numpy as np
import argparse
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path + '/flow_tool/')
import flowlib as fl

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="Path to kitti stereo dataset")
parser.add_argument("--pred_dir",    type=str, help="Path to the flow prediction")
args = parser.parse_args()

def main():
    img_num = 200
    noc_epe = np.zeros(img_num, dtype=np.float)
    noc_acc = np.zeros(img_num, dtype=np.float)
    occ_epe = np.zeros(img_num, dtype=np.float)
    occ_acc = np.zeros(img_num, dtype=np.float)

    eval_log = os.path.join(args.pred_dir, 'flow_result.txt')
    with open(eval_log, 'w') as el:
        for idx in range(img_num):
            # read groundtruth flow
            gt_noc_fn = args.dataset_dir + 'training/flow_noc/%.6d_10.png' % idx
            gt_occ_fn = args.dataset_dir + 'training/flow_occ/%.6d_10.png' % idx
            gt_noc_flow = fl.read_flow(gt_noc_fn)
            gt_occ_flow = fl.read_flow(gt_occ_fn)

            # read predicted flow (in png format)
            pred_flow_fn = args.pred_dir + 'png/%.6d.png' % idx
            pred_flow = fl.read_flow(pred_flow_fn)

            # resize pred_flow to the same size as gt_flow
            dst_h = gt_noc_flow.shape[0]
            dst_w = gt_noc_flow.shape[1]
            pred_flow = fl.resize_flow(pred_flow, dst_w, dst_h)

            # evaluation
            (single_noc_epe, single_noc_acc) = fl.evaluate_kitti_flow(gt_noc_flow, pred_flow, None)
            (single_occ_epe, single_occ_acc) = fl.evaluate_kitti_flow(gt_occ_flow, pred_flow, None)
            noc_epe[idx] = single_noc_epe
            noc_acc[idx] = single_noc_acc
            occ_epe[idx] = single_occ_epe
            occ_acc[idx] = single_occ_acc
            output_line = 'Flow %.6d Noc EPE = %.4f' + ' Noc ACC = %.4f' + ' Occ EPE = %.4f' + ' Occ ACC = %.4f\n';
            el.write(output_line % (idx, noc_epe[idx], noc_acc[idx], occ_epe[idx], occ_acc[idx]))

    noc_mean_epe = np.mean(noc_epe)
    noc_mean_acc = np.mean(noc_acc)
    occ_mean_epe = np.mean(occ_epe)
    occ_mean_acc = np.mean(occ_acc)

    print ('Mean Noc EPE = %.4f ' % noc_mean_epe)
    print ('Mean Noc ACC = %.4f ' % noc_mean_acc)
    print ('Mean Occ EPE = %.4f ' % occ_mean_epe)
    print ('Mean Occ ACC = %.4f ' % occ_mean_acc)

main()