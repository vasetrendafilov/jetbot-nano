"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 01.03.2022

Description: function library
             data operations: load, save, process
Python version: 3.6
"""

# python imports
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import random

import xml.etree.ElementTree as ET

# custom package imports
from Helpers_Localization import helper_postprocessing


def read_images_gram(path, size, depth, classes_list, shuffle):
    """
    load, resize, and normalize image data
    loads images from current folder
    :param path: global path of folder containing images of a data subset [string]
    :param size: output dimensions of the images (cols, rows) [tuple]
    :param depth: required depth of the loaded images (value: 1 or 3) [int]
    :return: array of normalized depth maps [ndarray]
    """

    images_list = []       # array of normalized images
    labels_list = []       # array of normalized images

    # list class folders
    for cl_index, cl in enumerate(classes_list):

        src_path = os.path.join(path, cl)

        for file_name in tqdm(os.listdir(src_path)):

            # skip non-image (BMP) files
            if not file_name[-4:] == '.bmp':
                continue

            if depth == 3:
                image = cv2.imread(os.path.join(src_path, file_name))
            else:
                image = cv2.imread(os.path.join(src_path, file_name), 0)

            if image is None:   # if file is read incorrectly
                continue

            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)   # bicubic interpolation

            image = image.reshape(image.shape[0], image.shape[1], depth)

            images_list.append(image)
            labels_list.append(cl_index)

    if len(images_list) == 0:
        print("No images were read.")
        exit(100)

    # shuffle data order
    if shuffle:
        data = list(zip(images_list, labels_list))
        random.shuffle(data)
        images_list, labels_list = zip(*data)

    images_list = np.array(images_list).astype(np.uint8)
    labels_list = np.array(labels_list).astype(np.uint8)

    return images_list, labels_list


def read_data_rpn(im_path, im_size, im_depth, annot_path, exclude_empty, shuffle):
    """
    load, resize, and normalize image data
    loads images and annotations from one folder
    :param im_path: global path of folder containing images of a data subset [string]
    :param im_size: output dimensions of the images (cols, rows) [tuple]
    :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
    :param shuffle: whether to shuffle input data order [bool]
    :return: images_list - array of normalized depth maps [ndarray]
             object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list]
    """

    images_list = []       # array of normalized images
    object_annotations_list = []       # array of array of bounding boxes for each image

    # list images in source folder
    for im_name in tqdm(os.listdir(os.path.join(im_path))):

        # --- load image ---
        if not im_name[-4:] != '.bmp':  # exclude system files
            continue

        if im_depth == 3:
            image = cv2.imread(os.path.join(im_path, im_name))
        else:
            image = cv2.imread(os.path.join(im_path, im_name), 0)

        if im_size != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)

        image = image.reshape(image.shape[0], image.shape[1], im_depth)

        # --- load annotations ---
        annot_name = im_name [:-4]+ '.csv'
        
        df = pd.read_csv(os.path.join(annot_path, annot_name))
        objects = []    # list of all objects in the image

        for _, row in df.iterrows():

            annot = [row['ymin'], row['xmin'], row['ymax'], row['xmax']]    # min_row, min_col, max_row, max_col

            if row['class'] == 'dash' and row['ymax'] - row['ymin'] + 1 > 5:  # select positive car samples, height > 25px
                objects.append(annot)

        if exclude_empty:
            if len(objects) > 0:
                images_list.append(image)
                object_annotations_list.append(objects)

        else:
            images_list.append(image)
            object_annotations_list.append(objects)

    if len(images_list) == 0:
        print("No images were read.")
        exit(100)

    if shuffle:
        data = list(zip(images_list, object_annotations_list))
        random.shuffle(data)
        images_list, object_annotations_list = zip(*data)

    images_list = np.array(images_list).astype(np.uint8)

    return images_list, object_annotations_list


def get_anchor_data_cls(bboxes, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio):
    """
    gnerate ground truth output for fully convolutional network for object detction
    single output branch, classifier only
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
    :param anchor_dims: tuple of anchor dimensions - (height, width) [tuple]
    :param img_dims: dimensions of input images (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param iou_low: samples with lower iou with all objects are declared negative (range: 0 to 1) [float]
    :param iou_high: samples with higher iou with an object are declared negative (range: 0 to 1) [float]
    :param num_negs_ratio: select negative samples num_negs_ratio times more than positive samples [int]
    :return: output_list - ground truth classes [ndarray]
             valid_inds - indices of images containing at least one object [list]
    """

    output_list = []    # ground truth
    valid_inds = []

    num_anchors = len(anchor_dims)

    for img_ind, img_bboxes in tqdm(enumerate(bboxes)):

        output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors + 1)
        output_class = np.zeros(output_dims_class).astype(np.int)

        # first position of an anchor center
        start_r = np.int(np.round(anchor_stride / 2))
        start_c = np.int(np.round(anchor_stride / 2))

        for output_row, center_row in enumerate(range(start_r, img_dims[0], anchor_stride)):  # iterate through rows of anchor centers
            for output_col, center_col in enumerate(range(start_c, img_dims[1], anchor_stride)):  # iterate through columns of anchor centers

                for anchor_ind, anchor_dim in enumerate(anchor_dims):  # iterate through different anchor sizes

                    for bbox in img_bboxes:  # iterate through annotated bounding boxes

                        # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
                        anchor = [max(0, center_row - np.int(anchor_dim[0]/2)),
                                  max(0, center_col - np.int(anchor_dim[1]/2)),
                                  min(center_row + np.int(anchor_dim[0]/2), img_dims[0]),
                                  min(center_col + np.int(anchor_dim[1]/2), img_dims[1])]
                        # min_row, min_col, max_row, max_col

                        iou = helper_postprocessing.calc_iou(bbox, anchor)

                        if iou >= iou_high:
                            # positive sample: set class, calculate deltas
                            output_class[output_row, output_col, anchor_ind] = 1

                        if (iou < iou_high) and (iou > iou_low):
                            # IOU between iou_min and iou_max
                            output_class[output_row, output_col, anchor_ind] = 2    # temporarily mark class with 2

        # mark negative samples
        for out_row in range(output_class.shape[0]):  # iterate through rows of output
            for out_col in range(output_class.shape[1]):  # iterate through columns of output

                if sum(output_class[out_row, out_col, :]) == 0:     # if no anchors at the specified center is marked with 1 (positive) or 2 (in-between)
                    output_class[out_row, out_col, -1] = 1

        # replace 2s with 0s
        output_class = np.where(output_class == 2, 0, output_class)

        # remove border pixels
        # make all centers where the anchor is partially out of the image
        for ind, anchor_dim in enumerate(anchor_dims):
            border_padding_h = np.int((anchor_dim[0] / anchor_stride) / 2) + 1
            border_padding_w = np.int((anchor_dim[1] / anchor_stride) / 2) + 1

            output_class[0:border_padding_h, :, ind] = 0
            output_class[output_class.shape[0] - border_padding_h:, :, ind] = 0
            output_class[:, 0:border_padding_w, ind] = 0
            output_class[:, output_class.shape[1] - border_padding_w:, ind] = 0

        # --- select negative samples ---
        # count positives and negatives
        num_positives = np.sum(output_class[:, :, 0:-1])

        # find negatives
        negs = output_class[:, :, -1]
        [r, c] = np.where(negs == 1)

        # select negatives to remove
        ind_to_remove = np.arange(len(r))
        np.random.shuffle(ind_to_remove)

        num_neg = min(len(r), num_positives * num_negs_ratio)   # number of positive to negative samples ratio: 1 to 10
        num_to_remove = len(r) - num_neg
        ind_to_remove = ind_to_remove[:num_to_remove]

        # remove negatives that were not selected
        for ind in ind_to_remove:
            output_class[r[ind], c[ind], :] = 0

        if num_positives > 0:
            output_list.append(output_class)
            valid_inds.append(img_ind)

    output_list = np.array(output_list)

    return output_list, valid_inds

	
def get_anchor_data(bboxes, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio):
    """
    generate ground truth output for fully convolutional network for object detection
    multi-output, classifier and regressor branch
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
    :param anchor_dims: tuple of anchor dimensions - (height, width) [tuple]
    :param img_dims: dimensions of input images (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param iou_low: samples with lower iou with all objects are declared negative (range: 0 to 1) [float]
    :param iou_high: samples with higher iou with an object are declared negative (range: 0 to 1) [float]
    :param num_negs_ratio: select negative samples num_negs_ratio times more than positive samples [int]
    :return: output_cls_arr - ground truth classes [ndarray]
             output_cls_arr - ground truth regression (normalized to [-1, 1]) [ndarray]
             valid_inds - indices of images containing at least one object [list]
             reg_norm_coef - normalization coefficient for ground truth regression data [float]
    """

    output_cls_list = []    # ground truth for classifier branch
    output_reg_list = []    # ground truth for regressor branch
    valid_inds = []

    num_anchors = len(anchor_dims)

    for img_ind, img_bboxes in tqdm(enumerate(bboxes)):

        output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors + 1)
        output_dims_reg = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors * 4)

        output_class = np.zeros(output_dims_class).astype(np.int)
        output_reg = np.zeros(output_dims_reg).astype(np.float32)

        # first position of an anchor center
        start_r = np.int(np.round(anchor_stride / 2))
        start_c = np.int(np.round(anchor_stride / 2))

        for output_row, center_row in enumerate(range(start_r, img_dims[0], anchor_stride)):  # iterate through rows of anchor centers
            for output_col, center_col in enumerate(range(start_c, img_dims[1], anchor_stride)):  # iterate through columns of anchor centers

                for anchor_ind, anchor_dim in enumerate(anchor_dims):  # iterate through different anchor sizes

                    for bbox in img_bboxes:  # iterate through annotated bounding boxes

                        # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
                        anchor = [max(0, center_row - np.int(anchor_dim[0]/2)),
                                  max(0, center_col - np.int(anchor_dim[1]/2)),
                                  min(center_row + np.int(anchor_dim[0]/2), img_dims[0]),
                                  min(center_col + np.int(anchor_dim[1]/2), img_dims[1])]
                        # min_row, min_col, max_row, max_col

                        iou = helper_postprocessing.calc_iou(bbox, anchor)

                        if iou >= iou_high:
                            # positive sample: set class, calculate deltas
                            output_class[output_row, output_col, anchor_ind] = 1

                            # set deltas - current location minus correct location
                            delta_r = bbox[0] - anchor[0]
                            delta_c = bbox[1] - anchor[1]
                            delta_h = np.int(bbox[2] - bbox[0]) - anchor_dim[0]
                            delta_w = np.int(bbox[3] - bbox[1]) - anchor_dim[1]

                            output_reg[output_row, output_col, anchor_ind * 4 + 0] = delta_r
                            output_reg[output_row, output_col, anchor_ind * 4 + 1] = delta_c
                            output_reg[output_row, output_col, anchor_ind * 4 + 2] = delta_h
                            output_reg[output_row, output_col, anchor_ind * 4 + 3] = delta_w

                        if (iou < iou_high) and (iou > iou_low):
                            # IOU between iou_min and iou_max
                            output_class[output_row, output_col, anchor_ind] = 2    # temporarily mark class with 2

                            # set deltas
                            # current location minus correct location
                            delta_r = bbox[0] - anchor[0]
                            delta_c = bbox[1] - anchor[1]
                            delta_h = np.int(bbox[2] - bbox[0]) - anchor_dim[0]
                            delta_w = np.int(bbox[3] - bbox[1]) - anchor_dim[1]

                            output_reg[output_row, output_col, anchor_ind * 4 + 0] = delta_r
                            output_reg[output_row, output_col, anchor_ind * 4 + 1] = delta_c
                            output_reg[output_row, output_col, anchor_ind * 4 + 2] = delta_h
                            output_reg[output_row, output_col, anchor_ind * 4 + 3] = delta_w

        # mark negative samples
        for out_row in range(output_class.shape[0]):  # iterate through rows of output
            for out_col in range(output_class.shape[1]):  # iterate through columns of output

                if sum(output_class[out_row, out_col, :]) == 0:     # if no anchors at the specified center is marked with 1 (positive) or 2 (in-between)
                    output_class[out_row, out_col, -1] = 1

        # replace 2s with 0s
        output_class = np.where(output_class == 2, 0, output_class)

        # remove border pixels
        # invalidate all centers where the anchor is partially out of the image
        # not applied to the class matrix of negative samples
        # regression ground truth for the negative samples is 0
        for ind, anchor_dim in enumerate(anchor_dims):
            border_padding_h = np.int((anchor_dim[0] / anchor_stride) / 2) + 1
            border_padding_w = np.int((anchor_dim[1] / anchor_stride) / 2) + 1

            output_class[0:border_padding_h, :, ind] = 0
            output_class[output_class.shape[0] - border_padding_h:, :, ind] = 0
            output_class[:, 0:border_padding_w, ind] = 0
            output_class[:, output_class.shape[1] - border_padding_w:, ind] = 0

            for num_anchor in range(4):
                # invalidate regression samples for all matrices for the specific anchor
                output_reg[0:border_padding_h, :, ind * 4 + num_anchor] = 0
                output_reg[output_reg.shape[0] - border_padding_h:, :, ind * 4 + num_anchor] = 0
                output_reg[:, 0:border_padding_w, ind * 4 + num_anchor] = 0
                output_reg[:, output_reg.shape[1] - border_padding_w:, ind * 4 + num_anchor] = 0


        # --- select negative samples ---
        # count positives and negatives
        num_positives = np.sum(output_class[:, :, 0:-1])

        # find negatives
        negs = output_class[:, :, -1]
        [r, c] = np.where(negs == 1)

        # select negatives to remove
        ind_to_remove = np.arange(len(r))
        np.random.shuffle(ind_to_remove)

        num_neg = min(len(r), num_positives * num_negs_ratio)   # number of positive to negative samples ratio: 1 to 10
        num_to_remove = len(r) - num_neg
        ind_to_remove = ind_to_remove[:num_to_remove]

        # remove negatives that were not selected
        for ind in ind_to_remove:
            output_class[r[ind], c[ind], :] = 0

        if num_positives > 0:
            output_cls_list.append(output_class)
            output_reg_list.append(output_reg)
            valid_inds.append(img_ind)

    output_cls_arr = np.array(output_cls_list)
    output_reg_arr = np.array(output_reg_list)

    reg_norm_coef = np.max(np.abs(output_reg_arr))
    output_reg_arr = output_reg_arr / reg_norm_coef   # normalize regression output to range same as the classifier output

    return output_cls_arr, output_reg_arr, valid_inds, reg_norm_coef


def save_results_cls_square(results_path, images, plot_color, output_cls, anchor_dims, anchor_stride, prob_thr):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_dims: tuple of tuples of anchor dimensions (height, width) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :return: None
    """

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images)):

        res = output_cls[im_ind, :, :, 0:-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)
        # d contains the indices of anchor size

        for pred_ind in range(len(r)):  # iterate over positive predictions

            anchor_dim = anchor_dims[d[pred_ind]]
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            min_row = np.int(center_row - np.round(anchor_dim[0] / 2))
            min_col = np.int(center_col - np.round(anchor_dim[1] / 2))

            # plot bounding box onto image
            cv2.rectangle(image, (min_col, min_row), (min_col + anchor_dim[1], min_row + anchor_dim[0]),
                              color=plot_color, thickness=1)

        # save test image with bounding boxes of detected objects
        cv2.imwrite(os.path.join(results_path, str(im_ind)+ '.bmp'), image)


def save_results(results_path, images, plot_color, output_cls, output_reg, anchor_dims, anchor_stride, prob_thr, norm_coef, output_branch):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_dims: tuple of tuples of anchor dimensions (height, width) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg = np.round(output_reg * norm_coef).astype(np.int)  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images)):

        res = output_cls[im_ind, :, :, 0:-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)
        # d contains the indices of anchor size

        for pred_ind in range(len(r)):  # iterate over positive predictions

            anchor_dim = anchor_dims[d[pred_ind]]
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 1]
            delta_h = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 2]
            delta_w = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 3]

            min_row = np.int(center_row - np.round(anchor_dim[0] / 2))
            min_col = np.int(center_col - np.round(anchor_dim[1] / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)
            h_adj = np.int(anchor_dim[0] + delta_h)
            w_adj = np.int(anchor_dim[1] + delta_w)

            max_row_adj = min_row_adj + h_adj
            max_col_adj = min_col_adj + w_adj

            # plot bounding box onto image
            if output_branch == 'classifier':
                cv2.rectangle(image, (min_col, min_row), (min_col + anchor_dim[1], min_row + anchor_dim[0]),
                              color=plot_color, thickness=1)

            elif output_branch == 'regressor':
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj),
                              color=plot_color, thickness=1)

            else:
                print("Invalid network output branch specified in function parameters.")
                exit(1)

        # save test image with bounding boxes of detected objects
        cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)
