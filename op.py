class Neuron(object):
    def __init__(self, input_neurons=[]):
        self.input_neurons = input_neurons
        self.output_neurons = []
        self.value = None

        for node in self.output_neurons:  # this node is output for each input
            node.output_nodes.append(self)

    def forward(self):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()


class Input(Neuron):
    def __init__(self):
        Neuron.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        pass


class Add(Neuron):
    def __init__(self, *inputs):
        Neuron.__init__(self, inputs)

    def forward(self):
        value = 0
        for neuron in self.input_neurons:
            value += neuron.value

        self.value = value

    def backward(self):
        pass


class Mul(Neuron):
    def __init__(self, *inputs):
        Neuron.__init__(self, *inputs)

    def forward(self):
        value = 1
        for neuron in self.input_neurons:
            value *= neuron.value

        self.value = value

    def backward(self):
        pass


class Perceptron(Neuron):
    def __init__(self, inputs, weights, bias):
        Neuron.__init__(self, inputs)
        self.weights = weights
        self.bias = bias

    def forward(self):
        value = self.bias.value
        for neuron, weight in zip(self.input_neurons, self.weights):
            value += weight.value * neuron.value

        self.value = value

    def backward(self):
        pass

