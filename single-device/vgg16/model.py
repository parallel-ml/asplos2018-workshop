""" This module creates VGG26 model. """
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model, Input


def vgg16():
    """
        Construct VGG16 model and return.
    """
    image = Input(224, 224, 3)

    # block 1
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(image)
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

    # block 2
    layer = Conv2D(128, (3, 3), activation='relu', padding='same')(layer)
    layer = Conv2D(128, (3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

    # block 3
    layer = Conv2D(256, (3, 3), activation='relu', padding='same')(layer)
    layer = Conv2D(256, (3, 3), activation='relu', padding='same')(layer)
    layer = Conv2D(256, (3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

    # block 4
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

    # block 5
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)

    # fully connected block
    layer = Flatten()(layer)
    layer = Dense(4096, activation='relu')(layer)
    layer = Dense(4096, activation='relu')(layer)
    layer = Dense(1000, activation='softmax')(layer)

    return Model(image, layer)
