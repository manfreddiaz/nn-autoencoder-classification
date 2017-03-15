from layers import *
import numpy as np
import cv2
from sklearn.utils import shuffle

BATCH_SIZE = 256


def preprocess(image):
    preprocessed = cv2.cvtColor(image.transpose(2, 1, 0), cv2.COLOR_RGB2GRAY)
    preprocessed = preprocessed.flatten()
    # preprocessed = preprocessed / np.linalg.norm(preprocessed)
    return preprocessed


def load_data():
    images = np.load("data/tinyX.npy")
    labels = np.load("data/tinyY.npy")

    proccessed = []
    for image in images:
        proccessed.append(preprocess(image))

    return np.array(proccessed, dtype=np.float32), labels


def train_autoencoder(images, hidden_units=200, epochs=20, learning_rate=1e-4):
    image_size = images.shape[1]

    x, y = Input(), Input()

    w1 = np.random.normal(0, 0.05, size=(image_size, hidden_units))
    b1 = np.zeros(hidden_units)

    w2 = np.random.normal(0, 0.05, size=(hidden_units, image_size))
    b2 = np.zeros(image_size)

    # encoder
    encoder = Dense(x, w1, b1, name='l1')
    encoder = Sigmoid(encoder, name='s1')

    # decoder
    decoder = Dense(encoder, w2, b2, name='l2')
    decoder = Sigmoid(decoder, name='s2')

    cost = MSE(y, decoder, name='ms1')

    computational_graph = compute_graph(cost)

    images = shuffle(images)
    for i in range(epochs):
        for index in range(0, len(images)): #, BATCH_SIZE):
            # batch = np.array(images[index: index + BATCH_SIZE])
            x.value = images[index]
            y.value = images[index]
            feed_forward_and_backward(computational_graph)
        stochastic_gradient_descent(computational_graph, learning_rate=learning_rate)

        print("Epoch: {}, Loss: {:.3f}".format(i, cost.value))

    return encoder.value

images, labels = load_data()
train_autoencoder(images)

