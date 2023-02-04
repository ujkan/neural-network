import unittest
import numpy as np
from src import neural_net


def process_training_files():
    labels_file = open("./data/train-labels-idx1-ubyte", "rb")
    labels_file.seek(8)
    images_file = open("./data/train-images-idx3-ubyte", "rb")
    images_file.seek(16)
    labels = []
    images = []
    for i in range(60000):
        label = int.from_bytes(labels_file.read(1), byteorder="big")
        labels.append(label)

        img = images_file.read(784)
        images.append([b/255 for b in img])
    labels_file.close()
    images_file.close()
    return images, labels

def process_test_files():
    labels_file = open("./data/t10k-labels-idx1-ubyte", "rb")
    labels_file.seek(8)
    images_file = open("./data/t10k-images-idx3-ubyte", "rb")
    images_file.seek(16)
    labels = []
    images = []
    for i in range(10000):
        label = int.from_bytes(labels_file.read(1), byteorder="big")
        labels.append(label)

        img = images_file.read(784)
        images.append([b/255 for b in img])
    labels_file.close()
    images_file.close()
    return images, labels


class TestMNIST(unittest.TestCase):

    def test(self):
        print("TEST STARTED")
        test_images, test_labels = process_test_files()
        net = neural_net.Net([764, 16, 16, 10])
        # Loading weights
        net.weights = np.load('weights/weights.npy', allow_pickle=True)
        net.biases = np.load('weights/biases.npy', allow_pickle=True)

        accuracy = net.test(test_images, test_labels)
        self.assertGreaterEqual(accuracy, 0.9)









