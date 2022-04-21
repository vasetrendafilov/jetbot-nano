"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 16.03.2022

Description: function library
             data postprocessing operations: non-maximum suppression
Python version: 3.6
"""

# python imports


def calc_iou(box1, box2):
    """

    :param box1: list of coordinates: row1, row2, col1, col2 [list]
    :param box2: list of coordinates: row1, row2, col1, col2 [list]
    :return: iou value
    """

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # respective area of the two boxes
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # overlap area
    interArea = max(xB - xA, 0) * max(yB - yA, 0)

    # IOU
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou
