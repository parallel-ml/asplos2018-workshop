"""
    This module defines different blocks in the VGG16 neural network.
"""
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model, Input


def block1():
    """ First Conv layer. """
    image = Input(shape=(224, 224, 3))
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(image)
    layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)
    model = Model(image, layer)
    return model


def block234():
    """ Middle Conv layer. """
    image = Input(shape=(112, 112, 64))

    # block 2
    layer = Conv2D(128, (3, 3), activation='relu', padding='same')(image)
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
    model = Model(image, layer)
    return model


def block5():
    """ Last Conv layer with flatten layer. """
    image = Input(shape=(14, 14, 512))
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(image)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)
    layer = Flatten()(layer)
    model = Model(image, layer)
    return model


def fc1():
    """ First fully connected layer in VGG16. """
    image = Input(shape=(25088,))
    layer = Dense(2048, activation='relu')(image)
    model = Model(image, layer)
    return model


def fc2():
    """ Last two fully connected layer. """
    image = Input(shape=(4096,))
    layer = Dense(4096, activation='relu')(image)
    layer = Dense(1000, activation='softmax')(layer)
    model = Model(image, layer)
    return model
