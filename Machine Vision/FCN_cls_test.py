"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.03.2021

Description: design, train and evaluate a CNN for object classification
Python version: 3.6
"""

# python imports
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.optimizers import Adam

# custom package imports
from Helpers_Localization import helper_model
from Helpers_Localization import helper_data
from Helpers_Localization import helper_losses


# --- paths ---
version = 'LV04_v10_3VGGblocks_slim_3anchors_extraConv_05_03_5_proba'

# NOTE: specify destination paths
srcImagesPath = r'D:\Science\Elena\MachineVision\Data\M-30\images_split'
srcAnnotationsPath = r'D:\Science\Elena\MachineVision\Data\M-30\GRAM-RTMv4\Annotations\M-30\xml'

srcModelPath = r'D:\Science\Elena\MachineVision\Models\LV04_v10_3VGGblocks_slim_3anchors_extraConv_05_03_5'

dstResultsPath = r'D:\Science\Elena\MachineVision\Results_tmp'
gtDstPath = r'D:\Science\Elena\MachineVision\GT'


# create folders to save data from the current execution
if not os.path.exists(os.path.join(dstResultsPath, version)):
    os.mkdir(os.path.join(dstResultsPath, version))
else:
    # to avoid overwriting training results
    print(f"Folder name {version} exists.")
    exit(1)

resultsPath = os.path.join(dstResultsPath, version)


# --- variables ---
class_names = ('bgr', 'cars')   # the element index marks the integer coding of classes (bgr - 0, cars - 1)

imgDims = {'rows': 480, 'cols': 800}    # input image dimensions
num_classes = 1
img_depth = 3
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

prob_thr = 0.5  # probability threshold
plot_color = (0, 255, 0)    # color of resulting bounding boxes


# --- load and format data ---
# load full dataset into memory - image data and labels
x_test, bboxes_test = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'test'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=False, shuffle=False)
print(f'Number of test samples: {x_test.shape[0]}')


# --- prepare ground truth data in required format ---
anchor_dims = (32, 64, 92)      # square anchors
anchor_stride = 8   # NOTE: depends on the model configuration

# --- generate ground truth output for teh test set ---
# iou_low = 0.3
# iou_high = 0.5
# y_class_test, valid_test = helper_data.get_anchor_data_cls(bboxes_test, anchor_dims, img_dims, anchor_stride, num_classes, iou_low, iou_high)


# --- construct model ---
lr = 0.0001
model = helper_model.load_model(model_path=os.path.join(srcModelPath, 'model.json'),
                                weights_path=os.path.join(srcModelPath, 'model.h5'))   # build model architecture

# compile model
model.compile(loss={
                  'rpn_out_class': helper_losses.rpn_loss_cls_multilabel,
                   },
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])

# --- convert test images to grayscale (CNN input format) ---
x_test_1 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in x_test]
x_test_1 = np.array(x_test_1)
x_test_1 = x_test_1.reshape(x_test_1.shape + (1,))

# --- apply model to test data ---
output_cls = model.predict(x_test_1, verbose=1)


# --- remove border pixels ---
'''
border_padding = np.int((anchor_dims[-1] / anchor_stride) / 2) + 1

output_cls[0:border_padding, :, :] = 0
output_cls[output_cls.shape[0] - border_padding:, :, :] = 0
output_cls[:, 0:border_padding, :] = 0
output_cls[:, output_cls.shape[1] - border_padding:, :] = 0
'''

# --- histogram of predicted probabilities ---
output_cls_pos_flat = output_cls[:, :, :, :-1].flatten()    # select objectness maps for positive objects only
plt.hist(output_cls_pos_flat, density=False, bins=100)  # density=False shows counts, True shows density
plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
plt.ylabel('Count')
plt.xlabel('Probability values')
plt.show()


# --- save results ---
helper_data.save_results_cls_square(resultsPath, x_test, plot_color, output_cls, anchor_dims, anchor_stride, prob_thr)
