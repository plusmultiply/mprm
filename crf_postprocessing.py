# Basic libs
import json
import os

import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply
from utils.mesh import rasterize_mesh

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir
import sys
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from sklearn.metrics import confusion_matrix
from utils.metrics import IoU_from_confusions
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



label_to_names = {0: 'unclassified',
                1: 'wall',
                2: 'floor',
                3: 'cabinet',
                4: 'bed',
                5: 'chair',
                6: 'sofa',
                7: 'table',
                8: 'door',
                9: 'window',
                10: 'bookshelf',
                11: 'picture',
                12: 'counter',
                14: 'desk',
                16: 'curtain',
                24: 'refridgerator',
                28: 'shower curtain',
                33: 'toilet',
                34: 'sink',
                36: 'bathtub',
                39: 'otherfurniture'}
num_classes = len(label_to_names)
label_values = np.sort([k for k, v in label_to_names.items()])
label_names = [label_to_names[k] for k in label_values]
label_to_idx = {l: i for i, l in enumerate(label_values)}
name_to_label = {v: k for k, v in label_to_names.items()}

def _create_pairwise_gaussian_3d(points, scale, shape):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 3
    feats = np.zeros((feat_size, shape), dtype=np.float32)
    for i in range(shape):
        feats[0, i] = points[i][0] / scale
        feats[1, i] = points[i][1] / scale
        feats[2, i] = points[i][2] / scale
    #print(np.shape(feats))
    return feats.reshape([feat_size, -1])

def _create_pairwise_bilateral_3d(spoint, srgb, points, rgb):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 6
    feats = np.zeros((feat_size, points.shape[0]), dtype=np.float32)
    for i in range(points.shape[0]):
        feats[0, i] = points[i][0] / spoint
        feats[1, i] = points[i][1]  / spoint
        feats[2, i] = points[i][2]  / spoint
        feats[3, i] = rgb[i][0] / srgb
        feats[4, i] = rgb[i][1] / srgb
        feats[5, i] = rgb[i][2] / srgb
    #print(np.shape(feats))
    return feats.reshape([feat_size, -1])

def _create_pairwise_bilateral_3d_coord(spoint, points):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 3
    feats = np.zeros((feat_size, points.shape[0]), dtype=np.float32)
    for i in range(points.shape[0]):
        feats[0, i] = points[i][0] / spoint
        feats[1, i] = points[i][1]  / spoint
        feats[2, i] = points[i][2]  / spoint
    return feats.reshape([feat_size, -1])

def crf_process(points, colors, preds, num_classes):
    d = dcrf.DenseCRF(points.shape[0], num_classes)
    #U = unary_from_labels(preds, num_classes, gt_prob=0.5, zero_unsure=False)
    U = unary_from_softmax(preds, scale=0.5)
    d.setUnaryEnergy(U)
    feats = _create_pairwise_gaussian_3d(points, 0.2, shape=points.shape[0])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = _create_pairwise_bilateral_3d(0.2, 0.04, points, colors)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    MAP = np.argmax(Q, axis=0)
    return MAP


if __name__ == '__main__':
    path = '/home/jiacheng/codes/KPconv/test/psa_mb_no_dual_no_dp_4dim_long/val_probs'

    files = listdir(path)
    pseudo_mask_path = '/mnt/sdc1/jiacheng/pseudo_mask/psa_mb_nodp_crf_4dim_80epoch/'
    num_classes = 20
    Confs = []
    Confs_ori = []

    crf = True
    save_result = True

    for file in tqdm(files):
        if file[-4:] == '.ply':
            cloud_name = file.split('/')[-1][:-4]
            data = read_ply(join(path, file))
            #data1 = read_ply(join(path1, file))
            points = np.vstack((data['x'], data['y'], data['z'])).T
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            # preds = data['preds']
            probs = np.vstack((data['wall'], data['floor'], data['cabinet'], data['bed'], data['chair'],
                               data['sofa'], data['table'], data['door'], data['window'], data['bookshelf'],
                               data['picture'], data['counter'], data['desk'], data['curtain'], data['refridgerator'],
                               data['shower_curtain'], data['toilet'], data['sink'], data['bathtub'],
                               data['otherfurniture'])).T
            probs = softmax(probs, axis=1)
            ori_preds = np.argmax(probs, axis=1) + 1
            ori_preds = label_values[ori_preds]
            gt = data['gt']
            # gt_ = np.unique(gt)
            # gt_ = np.array([label_to_idx[l] for l in np.unique(gt_)])
            # gt_ = gt_[gt_>0]-1
            # label_filter = np.zeros((1,20))
            # label_filter[0][gt_]=1
            #colors = np.vstack((data['red'], data['green'], data['blue'])).T
            # preds = np.array([label_to_idx[l] for l in preds])
            # gt = np.array([label_to_idx[l] for l in gt])
            # preds = preds -1
            # probs = probs * label_filterSe
            if crf:
                probs = np.swapaxes(probs, 0, 1)
                preds = crf_process(points, rgb, probs, num_classes)
                preds += 1
                preds = label_values[preds]
                Confs += [confusion_matrix(gt, preds, label_values)]
                if save_result:
                    val_name = join(pseudo_mask_path, cloud_name)

                    # Save file

                    write_ply(val_name,
                              [points, rgb, preds, gt],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'preds', 'class'])
                #val_name = join(pseudo_mask_path, cloud_name)
                #np.save(val_name + '.npy', preds)
            Confs_ori += [confusion_matrix(gt, ori_preds, label_values)]

            # save


    # Regroup confusions
    if crf:
        C = np.sum(np.stack(Confs), axis=0)
    OC = np.sum(np.stack(Confs_ori), axis=0)
    ignored_labels = [0]

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(label_values))):
        if label_value in ignored_labels:
            if crf:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)
            OC = np.delete(OC, l_ind, axis=0)
            OC = np.delete(OC, l_ind, axis=1)

    if crf:
        IoUs = IoU_from_confusions(C)
        mIoU = np.mean(IoUs)
        s = '{:5.2f} | '.format(100 * mIoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(100 * IoU)
        print('-' * len(s))
        print(s)
        print('-' * len(s) + '\n')

    IoUs = IoU_from_confusions(OC)
    mIoU = np.mean(IoUs)
    s = '{:5.1f} | '.format(100 * mIoU)
    for IoU in IoUs:
        s += '{:5.1f} '.format(100 * IoU)
    print('-' * len(s))
    print(s)
    print('-' * len(s) + '\n')