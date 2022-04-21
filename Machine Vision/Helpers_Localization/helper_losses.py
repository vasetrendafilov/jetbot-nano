"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 19.03.2021

Description: function library
             loss functions adjusted for fully convolutional classification and localization models
Python version: 3.6
"""

# python imports
from keras import backend as K
from keras.objectives import binary_crossentropy


# loss function weight coefficients
lambda_rpn_reg = 0.00001
lambda_rpn_class = 1.0

epsilon = 1e-4


def rpn_loss_cls_multilabel(y_true, y_pred):
    """
    calculate classification loss
    categorical cross-entropy adjusted for feature vectors containing all zeros
    :param y_true: ground truth output
    :param y_pred: predicted output
    :return: classification loss value
    """

    return K.sum(binary_crossentropy(y_true, y_pred), axis=(0, 1, 2)) / K.sum(
        K.cast(K.greater(binary_crossentropy(y_true, y_pred), 0), 'float32'), axis=(0, 1, 2))

		
def rpn_loss_reg(y_true, y_pred):
    """
    regression task loss function
    suitable for multi-hot encoded data
    :param y_true: ground truth output
    :param y_pred: predicted output
    :return: regression loss value (logloss)
    """

    mask = K.cast(K.not_equal(y_true, 0), 'float32')    # location of non-zero elements in ground truth

    dif = y_true - y_pred   # element-wise error
    dif = dif*mask      # ignored elements (zeros in ground truth) should not contribute to the loss value

    x_abs = K.abs(dif)
    x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

    return lambda_rpn_reg * K.sum(x_bool * (0.5 * dif * dif) + (1 - x_bool) * (x_abs - 0.5), axis=(1, 2, 3))
