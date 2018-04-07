

#!/usr/bin/python
"""
# ==============================
# flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""
import png
import pfm
import numpy as np
import matplotlib.colors as cl
# import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
import cv2

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

"""
=============
Flow Section
=============
"""


#def show_flow(filename):
#    """
#    visualize optical flow map using matplotlib
#    :param filename: optical flow file
#    :return: None
#    """
#    flow = read_flow(filename)
#    img = flow_to_image(flow)
#    plt.imshow(img)
#    plt.show()


# def visualize_flow(flow, mode='Y'):
#    """
#    this function visualize the input flow
#    :param flow: input flow in array
#    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
#    :return: None
#    """
#    if mode == 'Y':
#        # Ccbcr color wheel
#        img = flow_to_image(flow)
#        plt.imshow(img)
#        plt.show()
#    elif mode == 'RGB':
#        (h, w) = flow.shape[0:2]
#        du = flow[:, :, 0]
#        dv = flow[:, :, 1]
#        valid = flow[:, :, 2]
#        max_flow = max(np.max(du), np.max(dv))
#        img = np.zeros((h, w, 3), dtype=np.float64)
#        # angle layer
#        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
#        # magnitude layer, normalized to 1
#        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
#        # phase layer
#        img[:, :, 2] = 8 - img[:, :, 1]
#        # clip to [0,1]
#        small_idx = img[:, :, 0:3] < 0
#        large_idx = img[:, :, 0:3] > 1
#        img[small_idx] = 0
#        img[large_idx] = 1
#        # convert to rgb
#        img = cl.hsv_to_rgb(img)
#        # remove invalid point
#        img[:, :, 0] = img[:, :, 0] * valid
#        img[:, :, 1] = img[:, :, 1] * valid
#        img[:, :, 2] = img[:, :, 2] * valid
#        # show
#        plt.imshow(img)
#        plt.show()
#
#    return None


def read_flow(filename):
    """
    read optical flow data from flow file
    :param filename: name of the flow file
    :return: optical flow data in numpy array
    """
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)
    elif filename.endswith('.png'):
        # flow = read_png_file(filename)
        flow = read_kitti_png_file(filename)
    elif filename.endswith('.pfm'):
        flow = read_pfm_file(filename)
    else:
        raise Exception('Invalid flow file format!')

    return flow


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def save_flow_image(flow, image_file):
    """
    save flow visualization into image file
    :param flow: optical flow data
    :param flow_fil
    :return: None
    """
    # print flow.shape
    flow_img = flow_to_image(flow)
    img_out = Image.fromarray(flow_img)
    img_out.save(image_file)


def flowfile_to_imagefile(flow_file, image_file):
    """
    convert flowfile into image file
    :param flow: optical flow data
    :param flow_fil
    :return: None
    """
    flow = read_flow(flow_file)
    save_flow_image(flow, image_file)


def segment_flow(flow):
    h = flow.shape[0]
    w = flow.shape[1]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idx = ((abs(u) > LARGEFLOW) | (abs(v) > LARGEFLOW))
    idx2 = (abs(u) == SMALLFLOW)
    class0 = (v == 0) & (u == 0)
    u[idx2] = 0.00001
    tan_value = v / u

    class1 = (tan_value < 1) & (tan_value >= 0) & (u > 0) & (v >= 0)
    class2 = (tan_value >= 1) & (u >= 0) & (v >= 0)
    class3 = (tan_value < -1) & (u <= 0) & (v >= 0)
    class4 = (tan_value < 0) & (tan_value >= -1) & (u < 0) & (v >= 0)
    class8 = (tan_value >= -1) & (tan_value < 0) & (u > 0) & (v <= 0)
    class7 = (tan_value < -1) & (u >= 0) & (v <= 0)
    class6 = (tan_value >= 1) & (u <= 0) & (v <= 0)
    class5 = (tan_value >= 0) & (tan_value < 1) & (u < 0) & (v <= 0)

    seg = np.zeros((h, w))

    seg[class1] = 1
    seg[class2] = 2
    seg[class3] = 3
    seg[class4] = 4
    seg[class5] = 5
    seg[class6] = 6
    seg[class7] = 7
    seg[class8] = 8
    seg[class0] = 0
    seg[idx] = 0

    return seg


def flow_error(tu, tv, u, v):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]
    index_su = su[ind2]
    index_sv = sv[ind2]
    an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    '''
    angle = un * tun + vn * tvn + (an * tn)
    index = [angle == 1.0]
    angle[index] = 0.999
    ang = np.arccos(angle)
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    '''

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[ind2]
    mepe = np.mean(epe)
    return mepe

def flow_kitti_error(tu, tv, u, v, mask, ru = None, rv = None):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :param mask: ground-truth mask
    :return: End point error of the estimated flow
    """
    tau = [3, 0.05]
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]
    smask = mask[:]

    ind_valid = (smask != 0)
    n_total = np.sum(ind_valid)
    # print stu.size
    # print n_total

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    mag = np.sqrt(stu ** 2 + stv ** 2) + 1e-5

    epe = epe[ind_valid]
    mag = mag[ind_valid]

    if ru != None and rv != None:
        sru = ru[:]
        srv = rv[:]
        rig_mag = np.sqrt((stu - sru) ** 2 + (stv - srv) ** 2) + 1e-5
        rig_mag = rig_mag[ind_valid]

    phased_epe = []
    '''
    phased_error = range(0, 256, 5)
    for i in range(len(phased_error)-1):
        filter_mask = rig_mag < phased_error[i+1] #np.logical_and((mag > phased_error[i]), (mag < phased_error[i+1]))
        tmp_epe = np.mean(epe[filter_mask])
        phased_epe.append(tmp_epe)
    '''

    err = np.logical_and((epe > tau[0]), (epe / mag) > tau[1])
    n_err = np.sum(err)

    # print n_err
    # print n_total
    mean_epe = np.mean(epe)
    mean_acc = 1 - (float(n_err) / float(n_total))
    # print mean_epe
    # print mean_acc
    if ru != None and rv != None:
        return (mean_epe, mean_acc, phased_epe)
    else:
        return (mean_epe, mean_acc)


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    # print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def evaluate_flow_file(gt_file, pred_file):
    """
    evaluate the estimated optical flow end point error according to ground truth provided
    :param gt_file: ground truth file path
    :param pred_file: estimated optical flow file path
    :return: end point error, float32
    """
    # Read flow files and calculate the errors
    gt_flow = read_flow(gt_file)        # ground truth flow
    eva_flow = read_flow(pred_file)     # predicted flow
    # Calculate errors
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1], eva_flow[:, :, 0], eva_flow[:, :, 1])
    return average_pe


def evaluate_flow(gt_flow, pred_flow):
    """
    gt: ground-truth flow
    pred: estimated flow
    """
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1], pred_flow[:, :, 0], pred_flow[:, :, 1])
    return average_pe

def evaluate_kitti_flow(gt_flow, pred_flow, rigid_flow = None):
    # print gt_flow.shape
    if gt_flow.shape[2] == 2:
        gt_mask = np.ones((gt_flow.shape[0], gt_flow.shape[1]))
        epe, acc = flow_kitti_error(gt_flow[:, :, 0], gt_flow[:, :, 1], pred_flow[:, :, 0], pred_flow[:, :, 1], gt_mask)
        #epe, acc, phase = flow_kitti_error(gt_flow[:, :, 0], gt_flow[:, :, 1], pred_flow[:, :, 0], pred_flow[:, :, 1], gt_mask, rigid_flow[:,:,0], rigid_flow[:,:,1])
    elif gt_flow.shape[2] == 3:
        epe, acc = flow_kitti_error(gt_flow[:, :, 0], gt_flow[:, :, 1], pred_flow[:, :, 0], pred_flow[:, :, 1], gt_flow[:, :, 2])
        #epe, acc, phase = flow_kitti_error(gt_flow[:, :, 0], gt_flow[:, :, 1], pred_flow[:, :, 0], pred_flow[:, :, 1], gt_flow[:, :, 2], rigid_flow[:,:,0], rigid_flow[:,:,1])
    #return (epe, acc, phase)
    return (epe, acc)

"""
==============
Disparity Section
==============
"""


def read_disp_png(file_name):
    """
    Read optical flow from KITTI .png file
    :param file_name: name of the flow file
    :return: optical flow data in matrix
    """
    image_object = png.Reader(filename=file_name)
    image_direct = image_object.asDirect()
    image_data = list(image_direct[2])
    (w, h) = image_direct[3]['size']
    channel = len(image_data[0]) / w
    flow = np.zeros((h, w, channel), dtype=np.uint16)
    for i in range(len(image_data)):
        for j in range(channel):
            flow[i, :, j] = image_data[i][j::channel]
    return flow[:, :, 0] / 256


def disp_to_flowfile(disp, filename):
    """
    Read KITTI disparity file in png format
    :param disp: disparity matrix
    :param filename: the flow file name to save
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = disp.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    empty_map = np.zeros((height, width), dtype=np.float32)
    data = np.dstack((disp, empty_map))
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data.tofile(f)
    f.close()


"""
==============
Image Section
==============
"""


def read_image(filename):
    """
    Read normal image of any format
    :param filename: name of the image file
    :return: image data in matrix uint8 type
    """
    img = Image.open(filename)
    im = np.array(img)
    return im


def warp_image(im, flow):
    """
    Use optical flow to warp image to the next
    :param im: image to warp
    :param flow: optical flow
    :return: warped image
    """
    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    # fx = fx.astype('float32')
    # fy = fy.astype('float32')
    flow = np.rint(flow).astype('int32')
    fx += flow[:,:,0]
    fy += flow[:,:,1]
    mask = np.logical_or(fx <0 , fx > flow_width)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy > flow_height)
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
    warp = np.zeros((image_height, image_width, im.shape[2]))
    for i in range(im.shape[2]):
        channel = im[:, :, i]
        # plt.imshow(channel, cmap='gray')
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :, i] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)


"""
==============
Others
==============
"""

def pfm_to_flo(pfm_file):
    flow_filename = pfm_file[0:pfm_file.find('.pfm')] + '.flo'
    (data, scale) = pfm.readPFM(pfm_file)
    flow = data[:, :, 0:2]
    write_flow(flow, flow_filename)


def scale_image(image, new_range):
    """
    Linearly scale the image into desired range
    :param image: input image
    :param new_range: the new range to be aligned
    :return: image normalized in new range
    """
    min_val = np.min(image).astype(np.float32)
    max_val = np.max(image).astype(np.float32)
    min_val_new = np.array(min(new_range), dtype=np.float32)
    max_val_new = np.array(max(new_range), dtype=np.float32)
    scaled_image = (image - min_val) / (max_val - min_val) * (max_val_new - min_val_new) + min_val_new
    return scaled_image.astype(np.uint8)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print 'Magic number incorrect. Invalid .flo file'
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print "Reading %d x %d flow file in .flo format" % (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def read_png_file(flow_file):
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    print "Reading %d x %d flow file in .png format" % (h, w)
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow

def read_kitti_png_file(flow_file):
    # print flow_file
    flow_img = cv2.imread(flow_file, cv2.CV_LOAD_IMAGE_UNCHANGED)
    flow_img = flow_img.astype(float)
    # print flow_img.shape
    flow_data = np.zeros(flow_img.shape, dtype = np.float)
    flow_data[:, :, 0] = (flow_img[:, :, 2] - 2 ** 15) / 64.0
    flow_data[:, :, 1] = (flow_img[:, :, 1] - 2 ** 15) / 64.0
    flow_data[:, :, 2] = flow_img[:, :, 0]
    return flow_data

def read_pfm_file(flow_file):
    """
    Read from .pfm file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    import pfm
    (data, scale) = pfm.readPFM(flow_file)
    return data

def resize_flow(flow, des_width, des_height):
    src_height  = flow.shape[0]
    src_width   = flow.shape[1]
    ratio_height    = float(des_height) / float(src_height)
    ratio_width     = float(des_width) / float(src_width)
    # print ratio_width
    # print ratio_height
    flow = cv2.resize(flow, (des_width, des_height), interpolation=cv2.INTER_NEAREST)
    flow[:, :, 0] = flow[:, :, 0] * ratio_width
    flow[:, : ,1] = flow[:, :, 1] * ratio_height
    return flow

def remove_ambiguity_flow(flow_img, err_img, threshold_err = 10.0):
    thre_flow   = flow_img
    mask_img    = np.ones(err_img.shape, dtype = np.uint8)
    mask_img[err_img > threshold_err] = 0.0
    thre_flow[err_img > threshold_err] = 0.0
    return (thre_flow, mask_img)

def write_kitti_png_file(flow_fn, flow_data, mask_data):
    flow_img = np.zeros((flow_data.shape[0], flow_data.shape[1], 3), dtype = np.uint16)
    flow_img[:, :, 2] = flow_data[:, :, 0] * 64.0 + 2 ** 15
    flow_img[:, :, 1] = flow_data[:, :, 1] * 64.0 + 2 ** 15
    flow_img[:, :, 0] = mask_data[:, :]
    cv2.imwrite(flow_fn, flow_img)

def flow_kitti_mask_error(tu, tv, gt_mask, u, v, pd_mask):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param gt_mask: ground-truth mask

    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :param pd_mask: estimated flow mask
    :return: End point error of the estimated flow
    """
    tau = [3, 0.05]
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]
    s_gt_mask = gt_mask[:]
    s_pd_mask = pd_mask[:]

    ind_valid = np.logical_and(s_gt_mask != 0, s_pd_mask != 0)
    n_total = np.sum(ind_valid)
    # print stu.size
    # print n_total

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    mag = np.sqrt(stu ** 2 + stv ** 2) + 1e-5

    epe = epe[ind_valid]
    mag = mag[ind_valid]

    err = np.logical_and((epe > tau[0]), (epe / mag) > tau[1])
    n_err = np.sum(err)

    # print n_err
    # print n_total
    mean_epe = np.mean(epe)
    mean_acc = 1 - (float(n_err) / float(n_total))
    # print mean_epe
    # print mean_acc
    return (mean_epe, mean_acc)

