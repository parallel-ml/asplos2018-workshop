from model import vgg16
import numpy as np


def main():
    """ Call model construction function and run model multiple times. """
    model = vgg16()
    test_x = np.random.rand(224, 224, 3)
    for _ in range(50):
        model.predict(np.array([test_x]))


if __name__ == '__main__':
    main()
