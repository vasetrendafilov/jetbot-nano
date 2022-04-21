"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 01.03.2022

Description: function library
             training process monitoring and result statistics
Python version: 3.6
"""

# python imports
import matplotlib.pyplot as plt
import os
import numpy as np


def save_training_logs(history, dst_path):
    """
    saves graphs for the loss and accuracy of both the training and validation dataset throughout the epochs for comparison
    :param history: Keras callback object which stores accuracy information in each epoch [Keras history object]
    :param dst_path: destination for the graph images
    :return: None
    """

    # --- save accuracy graphs of training and validation sets ---
    plt.plot(history.history['accuracy'], 'r')
    plt.plot(history.history['val_accuracy'], 'g')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()
    # plt.show()    # blocks execution until figure is closed
    plt.savefig(os.path.join(dst_path, 'acc.png'))      # acc.png - name of accuracy graph
    plt.close()

    # --- save loss graphs of training and validation sets ---
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    # plt.show()    # blocks execution until figure is closed
    plt.savefig(os.path.join(dst_path, 'loss.png'))     # loss.png - name of loss graph
    plt.close()

    # --- save loss and accuracy of training and validation sets as a txt file ---
    losses = np.column_stack((history.history['loss'], history.history['val_loss']))
    np.savetxt(os.path.join(dst_path, 'loss.txt'), losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

    accuracies = np.column_stack((history.history['accuracy'], history.history['val_accuracy']))
    np.savetxt(os.path.join(dst_path, 'acc.txt'), accuracies, fmt='%.4f', delimiter='\t', header="TRAIN_ACC\tVAL_ACC")


def save_training_logs_ssd(history, dst_path):
    """
    saves graphs for the loss and accuracy of both the training and validation dataset throughout the epochs for comparison
    :param history: Keras callback object which stores accuracy information in each epoch [Keras history object]
    :param dst_path: destination for the graph images
    :return: None
    """

    # --- save combined loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
    plt.title('Combined loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'joint_loss.png'))
    plt.close()

    # --- save classification loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['rpn_out_class_loss'], 'r')
    plt.plot(history.history['val_rpn_out_class_loss'], 'g')
    plt.title('Classification loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_class', 'val_class'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'classification_loss.png'))
    plt.close()

    # --- save regression loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['rpn_out_reg_loss'], 'r')
    plt.plot(history.history['val_rpn_out_reg_loss'], 'g')
    plt.title('Regression loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_reg', 'val_reg'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'regression_loss.png'))
    plt.close()

    # --- save losses of training and validation sets as txt files ---
    joint_losses = np.column_stack((history.history['loss'], history.history['val_loss']))
    np.savetxt(os.path.join(dst_path, 'joint_loss.txt'), joint_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

    class_losses = np.column_stack((history.history['rpn_out_class_loss'], history.history['val_rpn_out_class_loss']))
    np.savetxt(os.path.join(dst_path, 'classification_loss.txt'), class_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

    reg_losses = np.column_stack((history.history['rpn_out_reg_loss'], history.history['val_rpn_out_reg_loss']))
    np.savetxt(os.path.join(dst_path, 'regression_loss.txt'), reg_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

	
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

    return fig
