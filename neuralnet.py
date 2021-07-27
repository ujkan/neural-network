# 28x28 = 784 inputs
# 2 hidden layers a 16 neurons
# 10 outputs for each digit (0-9)

import math
import numpy as np
NCONST=100
class Neuron:
    def __init__(self, value, fwdlinks, backlinks):
        self.value = value
        self.fwdlinks = fwdlinks
        self.backlinks = backlinks
    def setFwdLinks(self, links):
        self.fwdlinks = links
    def setBacklinks(self, links):
        self.backlinks = links

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class Net:
    def __init__(self, layers):
        self.layers = layers
        self.biases = [np.random.randn(len(k), 1) for k in layers[1:]]
        self.weights = [np.random.randn(len(k), len(l)) for l, k in zip(layers[0:-1], layers[1:])]

    def addNeuron(self, i, neuron):
        self.layers[i].append(neuron)

    def forward(self):
        for i in range(len(self.layers) - 1):
            nvals = []
            print("---")
            print(self.weights[i].shape)
            print(self.biases[i].shape)
            vals = np.array([[n.value] for n in self.layers[i]])
            print(vals.shape)
            if (i == 1):
                # print([n.value for n in self.layers[i]])
                print(len(self.layers[i]))
            nvals = np.matmul(self.weights[i],  vals) + self.biases[i]
            print(nvals.shape)
            for j in range(len(self.layers[i + 1])):
                self.layers[i + 1][j].value = nvals[j][0]
    def backpropagate(self):

    # a[n+1][i] = w[0][i] * a[n][0] + ... + w[k][i] * a[n][k] + b[i]
    # --> derivative wrt to w[j][i] is a[n][i]
    # --> derivative wrt to a[n][j] is w[j][i]
    # --> derivative wrt to b[i] is 1
    # so, update: w[j][i] := w[j][i] - step_size * a[n][i] (we want -GRAD, hence minus)
    #             a[n][j] := a[n][j] - step_size * w[j][i]
    #             b[i] := b[i] - step_size




def main():
    firstLayer = []
    for i in range(784):
        firstLayer.append(Neuron(0, None, None))

    secondLayer = []
    for i in range(16):
        secondLayer.append(Neuron(0, None, firstLayer))
    for i in range(784):
        firstLayer[i].setFwdLinks(secondLayer)

    thirdLayer = []
    for i in range(16):
        thirdLayer.append(Neuron(0, None, secondLayer))
    for i in range(16):
        secondLayer[i].setFwdLinks(thirdLayer)

    fourthLayer = []
    for i in range(10):
        fourthLayer.append(Neuron(0, None, thirdLayer))
    print("hi")
    net = Net([firstLayer, secondLayer, thirdLayer, fourthLayer])
    train(net)


def errSq(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i])**2
    return s

def train(net):
    labels, images = processFiles()
    errsum = 0
    for i in range(NCONST):
        for j in range(784):
            (net.layers[0])[j].value = images[i][j]
        net.forward()
        obtained = [n.value for n in net.layers[-1]]
        goal = [0] * 10
        goal[labels[i]] = 1
        errsum += errSq(obtained, goal)





def processFiles():
    labelsFile = open("/Users/Irfan/Downloads/train-labels-idx1-ubyte", "rb")
    labelsFile.seek(8)
    imagesFile = open("/Users/Irfan/Downloads/train-images-idx3-ubyte", "rb")
    imagesFile.seek(16)
    labels = []
    images = []
    for i in range(NCONST):
        label = int.from_bytes(labelsFile.read(1), byteorder="big")
        if (i < 3):
            print(label)
        labels.append(label)

        img = imagesFile.read(784)
        images.append([b for b in img])

    return labels, images


main()
