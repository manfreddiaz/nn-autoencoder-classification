import numpy as np


class Layer(object):
    def __init__(self, input_layer=[], name=None):
        self.name = name
        self.input_layers = input_layer
        self.output_layers = []
        self.value = None
        for layer in self.input_layers:
            layer.output_layers.append(self)
        self.gradients = {}
        self.trainables = []

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Input(Layer):
    def __init__(self, value=None, name=None):
        Layer.__init__(self, name=name)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.output_layers:
            gradient_value = n.gradients[self]
            self.gradients[self] += gradient_value * 1


class Dense(Layer):
    def __init__(self, input_layer, weights, bias, name=None):
        Layer.__init__(self, [input_layer, Input(weights), Input(bias)], name=name)
        self.W = self.input_layers[1]
        self.b = self.input_layers[2]
        self.trainables = [self.W, self.b]

    def forward(self):
        input_layer, weights, bias = self.input_layers
        self.value = np.dot(input_layer.value, weights.value) + bias.value

    def backward(self):
        self.gradients = {neuron: np.zeros_like(neuron.value) for neuron in self.input_layers}
        x = self.input_layers[0]
        W = self.input_layers[1]
        b = self.input_layers[2]
        for neuron in self.output_layers:
            gradient_value = neuron.gradients[self]
            self.gradients[x] += np.dot(gradient_value, W.value.T)
            self.gradients[W] += np.dot(x.value.T, gradient_value)
            self.gradients[b] += np.sum(gradient_value, axis=0, keepdims=False)


class Sigmoid(Layer):
    def __init__(self, input_layer, name=None):
        Layer.__init__(self, [input_layer], name=name)

    @staticmethod
    def _compute(x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        self.value = Sigmoid._compute(self.input_layers[0].value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.input_layers}
        input_layer = self.input_layers[0]
        for neuron in self.output_layers:
            gradient_value = neuron.gradients[self]
            self.gradients[input_layer] += self.value * (1 - self.value) * gradient_value


class MSE(Layer):
    def __init__(self, y, layer, name=None):
        Layer.__init__(self, [y, layer], name=name)
        self.error = None

    def forward(self):
        y = self.input_layers[0].value
        layer = self.input_layers[1].value
        self.error = y - layer
        self.value = np.mean(self.error ** 2)

    def backward(self):
        y = self.input_layers[0]
        layer = self.input_layers[1]
        size = y.value.shape[0]

        self.gradients[y] = (-2 / size) * self.error
        self.gradients[layer] = (-2 / size) * self.error


def feed_forward_and_backward(computational_graph):
    for layer in computational_graph:
        layer.forward()

    for layer in computational_graph[::-1]:
        layer.backward()


def stochastic_gradient_descent(computational_graph, learning_rate=1e-2):
    train_layers = []
    for layer in computational_graph:
        if layer.trainables and len(layer.trainables):
            train_layers.append(layer)

    if len(train_layers) > 0:
        for layer in train_layers:
            for trainable in layer.trainables:
                trainable.value -= learning_rate * layer.gradients[trainable]


def _flatten_layers(layer):
    layers = [layer]
    index = 0

    while index < len(layers):
        layers.extend(layers[index].input_layers)
        index += 1

    return layers

def compute_graph(layer):
    input_layers = [n for n in filter(lambda x: isinstance(x, Input), _flatten_layers(layer))]
    graph = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in graph:
            graph[n] = {'in': set(), 'out': set()}
        for m in n.output_layers:
            if m not in graph:
                graph[m] = {'in': set(), 'out': set()}
            graph[n]['out'].add(m)
            graph[m]['in'].add(n)
            layers.append(m)

    ordered_layers = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        ordered_layers.append(n)
        for m in n.output_layers:
            graph[n]['out'].remove(m)
            graph[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(graph[m]['in']) == 0:
                S.add(m)
    return ordered_layers