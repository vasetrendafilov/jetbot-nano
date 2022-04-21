"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 23.03.2022

Description: design, train and evaluate a CNN for object classification
Python version: 3.6
"""

# python imports
import os
import numpy as np

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# custom package imports
from Helpers_Localization import helper_model
from Helpers_Localization import helper_data
from Helpers_Localization import helper_stats
from Helpers_Localization import helper_losses


# --- paths ---
version = 'LV04_v1'

# NOTE: specify destination paths
srcImagesPath = r'D:\Science\Elena\MachineVision\Data\M-30\images_split'

srcAnnotationsPath = r'D:\Science\Elena\MachineVision\Data\M-30\GRAM-RTMv4\Annotations\M-30\xml'
dstModelsPath = r'D:\Science\Elena\MachineVision\Models_tmp'
gtDstPath = r'D:\Science\Elena\MachineVision\GT'

# create folders to save data from the current execution
if not os.path.exists(os.path.join(dstModelsPath, version)):
    os.mkdir(os.path.join(dstModelsPath, version))
else:
    # to avoid overwriting training results
    print(f"Folder name {version} exists.")
    exit(1)

modelsPath = os.path.join(dstModelsPath, version)


# --- variables ---
class_names = ('bgr', 'cars')   # the element index marks the integer coding of classes (bgr - 0, cars - 1)

imgDims = {'rows': 480, 'cols': 800}    # input image dimensions
num_classes = 1
img_depth = 1
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)


# --- load and format data ---
# load full dataset into memory - image data and ground truth bounding boxes
x_train_orig, bboxes_train = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'train'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=True, shuffle=True)
x_val_orig, bboxes_val = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'val'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=True, shuffle=True)
# x_test, bboxes_test = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'test'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=False, shuffle=False)

print(f'Training dataset shape: {x_train_orig.shape}')
print(f'Number of training samples: {x_train_orig.shape[0]}')
print(f'Number of validation samples: {x_val_orig.shape[0]}')
# print(f'Number of test samples: {x_test.shape[0]}')


# --- prepare ground truth data in required CNN output format ---
anchor_dims = (32, 48, 64, 92)   # square anchors
anchor_stride = 16      # NOTE: depends on the model configuration

# iou thresholds for positive and negative samples
iou_low = 0.3
iou_high = 0.7
num_negs_ratio = 10     # select X times more negative than positive samples

# --- form output matrices ---
# NOTE: images containing no objects, or objects which are not fully encased in an anchor, are discarded
y_class_train, valid_train = helper_data.get_anchor_data_cls(bboxes_train, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio)
y_class_val, valid_val = helper_data.get_anchor_data_cls(bboxes_val, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio)
# y_class_test, valid_test = helper_data.get_anchor_data_cls(bboxes_test, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio)

print(f'Ground truth of training set shape: {y_class_train.shape}')
print(f'Ground truth of validation set shape: {y_class_val.shape}')

print(f'Number of positive samples in training set: {np.sum(y_class_train[:, :, :, -1])}')
print(f'Number of positive samples in validation set: {np.sum(y_class_val[:, :, :, -1])}')


# --- remove images containing no objects ---
x_train = []
for valid_ind in valid_train:
    x_train.append(x_train_orig[valid_ind])
x_train = np.array(x_train)

x_val = []
for valid_ind in valid_val:
    x_val.append(x_val_orig[valid_ind])
x_val = np.array(x_val)


# --- save anchor data (creation takes a long time, save for same anchor sizes and output size) ---
# np.savetxt(os.path.join(gtDstPath, 'y_class_train.txt'), y_class_train.flatten(), fmt='%f')
# np.savetxt(os.path.join(gtDstPath, 'y_class_val.txt'), y_class_val.flatten(), fmt='%f')
# np.savetxt(os.path.join(gtDstPath, 'y_class_test.txt'), y_class_test.flatten(), fmt='%f')


# --- construct model ---
# optimization hyperprameters
epochs = 30
lr = 0.0001
batch_size = 25      # number of samples to process before updating the weights

model = helper_model.construct_model_ssd_cls(input_shape=img_dims, num_anchors=len(anchor_dims))   # build model architecture

# compile model
model.compile(loss={
                  'rpn_out_class': helper_losses.rpn_loss_cls_multilabel,     # loss function to be applied to the output layer named rpn_out_class
                   },
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])


# --- fit model ---
model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{val_accuracy:.4f}.hdf5'),   # epoch number and val accuracy will be part of the weight file name
                                   monitor='val_accuracy',      # metric to monitor when selecting weight checkpoints to save
                                   verbose=1,
                                   save_best_only=False)     # True saves only the weights after epochs where the monitored value (val accuracy) is improved

history = model.fit(x_train, y_class_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[model_checkpoint],
                    verbose=1,
                    validation_data=(x_val, y_class_val),
                    shuffle=True)


# --- save model ---
# save model architecture
print(model.summary())      # parameter info for each layer
with open(os.path.join(modelsPath, 'modelSummary.txt'), 'w') as fh:     # save model summary
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)   # save diagram of model architecture

# save model configuration and weights
model_json = model.to_json()  # serialize model architecture to JSON
with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5
print("Saved model to disk.")


# --- save training curves and logs ---
helper_stats.save_training_logs(history=history, dst_path=modelsPath)

# --- apply model to test data ---
# output_cls = model.predict(x_test, verbose=1)
