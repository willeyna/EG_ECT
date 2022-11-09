import numpy as np
from keras.datasets import mnist
import networkx as nx
import simplex_ect
import ect


# Creates small sample from MNIST dataset converted to binary images
# tol is intensity percentage at which to assign 1; 0 gives best "chunky" letters with surrounded holes for ECT
def generate_binary_MNIST(N=70000, tol = 0, loc = './', seed = None, savefile = True):

    tol = (tol*255)//1
    # getting MNIST in nice np format through keras
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    X = np.concatenate([train_X, test_X])
    Y = np.concatenate([train_y, test_y])
    # permute elements of MNIST
    np.random.seed(seed)
    p = np.random.permutation(len(X))
    X = X[p]
    Y = Y[p]

    IM = np.zeros([N, 28, 28])
    for i in range(N):
        IM[i] = X[i]
        IM[i][IM[i] <= tol] = 0
        IM[i][IM[i] > tol] = 1

    labels = Y[:N]

    if savefile:
        if N == 70000:
            # different file name if full MNIST used
            np.save(f'{loc}binary_MNIST.npy', IM)
            np.save(f'{loc}binary_MNIST_labels.npy', labels)
        else:
            np.save(f'{loc}binary_MNIST_{N}.npy', IM)
            np.save(f'{loc}binary_MNIST_{N}_labels.npy', labels)

    return IM, labels

## THESE TWO NEED TESTING AND CLEANING
## (better solution than storing labels in different file than graph?, list[3]?)

# takes in a generated binary MNIST dataset and performs complexification, saving the graph list structure w/ pickle
def generate_graph_MNIST(binary_MNIST, name = 'graph_MNIST.npy', loc = './', savefile = True):
    N = len(binary_MNIST)
    MNISTCells = []
    for i in range(N):
        print(f'Complexification {round(i/N * 100, 1)}% complete', end = '\r')
        MNISTCells.append(ECT.complexify(binary_MNIST[i], center=False))

    if savefile:
        np.save(loc + name, MNISTCells, allow_pickle=True)

    return MNISTCells

def generate_MNIST_ECT(graph_MNIST, MNIST_labels = None, name = 'MNIST_ECT.npy',
                            loc = './', angles = [0, np.pi/2, np.pi, 3*np.pi/4], radial = False, T = 16, savefile = True):
    N = len(graph_MNIST)
    ect = np.zeros([N, T*(len(angles) + radial)])
    for i in range(N):
        print(f'ECT computation {round(i/N * 100, 1)}% complete', end = '\r')

        dECT = ECT.directional_ect(graph_MNIST[i], T=T, angles = angles)
        if radial:
            rECT = ECT.radial_ect(graph_MNIST[i], T=T)
            ect[i] = np.concatenate([dECT, rECT])
        else:
            ect[i] = dECT

    if MNIST_labels is not None:
        MNIST_ECT = np.column_stack([MNIST_labels, ect])
    else:
        MNIST_ECT = ect

    if savefile:
        np.save(loc + name, MNIST_ECT, allow_pickle=True)

    return MNIST_ECT
