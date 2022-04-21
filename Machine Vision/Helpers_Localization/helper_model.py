"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 16.03.2022

Description: function library
             model operations: construction, loading, saving
Python version: 3.6
"""

# python imports
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input
from keras.models import Sequential, Model, model_from_json


def load_model(model_path, weights_path):
    """
    loads a pre-trained model configuration and calculated weights
    :param model_path: path of the serialized model configuration file (.json) [string]
    :param weights_path: path of the serialized model weights file (.h5) [string]
    :return: model - keras model object
    """

    # --- load model configuration ---
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)     # load model architecture

    model.load_weights(weights_path)     # load weights

    return model


def construct_model_cnn_gram(num_classes, input_shape):
    """
    construct model architecture
    :param num_classes: number of output classes of the model [int]
    :return: model - Keras model object
    """

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def construct_model_ssd_cls(input_shape, num_anchors):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """

    input_layer = Input(shape=input_shape)

    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(input_layer)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f3)
    f5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f4)
    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    # classifier
    x_class_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                       kernel_initializer='he_normal')(f6)
    x_class_2 = Conv2D(filters=num_anchors + 1, kernel_size=(1, 1), padding='same', activation='sigmoid',
                       kernel_initializer='glorot_uniform', name="rpn_out_class")(x_class_1)

    model = Model(input_layer, x_class_2)

    return model
	

def construct_model_ssd(input_shape, num_anchors):
    """
    construct region proposal model architecture
    classifier architecture for binary classification
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :param num_anchors: number of different anchors with the same center [int]
    :return:  model - Keras model object
    """

    input_layer = Input(shape=input_shape)

    # feature extractor
    f1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
        input_layer)
    f2 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f1)
    f3 = MaxPool2D(pool_size=(2, 2))(f2)

    f4 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f3)
    f5 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f4)
    f6 = MaxPool2D(pool_size=(2, 2))(f5)

    f7 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f6)
    f8 = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f7)
    f9 = MaxPool2D(pool_size=(2, 2))(f8)

    # classifier
    x_class_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                       kernel_initializer='he_normal')(f9)
    x_class_2 = Conv2D(filters=num_anchors + 1, kernel_size=(1, 1), padding='same', activation='sigmoid',
                       kernel_initializer='glorot_uniform', name="rpn_out_class")(x_class_1)

    # regressor
    x_reg_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(f9)
    x_reg_2 = Conv2D(filters=num_anchors * 4, kernel_size=(1, 1), padding='same', activation='linear',
                     kernel_initializer='zeros', name="rpn_out_reg")(x_reg_1)

    model = Model(input_layer, [x_class_2, x_reg_2])

    return model
