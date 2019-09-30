import Neuron as neu
import Parser as files


class Network:

    def __init__(self, n_inputs, n_hidden_layers, n_outputs, if_bias, learn_coef, momentum_coef):

        # parameters:
        #   n_inputs - number of inputs
        #   n_hidden_layers - number of neurons in hidden layer
        #   n_outputs - number of desired outputs
        #   ifBias - decide if bias is added to calculation
        #   learn_coef - eta, learing coefficient
        #   momentum_coef - alfa, momentum coefficient
        # Lists containing all neurons in the layers
        self.hiddenLayer = []
        self.outputLayer = []

        neu.NeuronHidden.sigmoid_f = neu.sigmoid
        neu.NeuronHidden.dF = neu.sigmoid_derivative
        neu.NeuronOutput.sigmoid_f = neu.sigmoid
        neu.NeuronOutput.dF = neu.sigmoid_derivative

        # Initializing desired number of neurons in HL and OL, applying incrementing indexes to HL neurons
        for i in range(n_hidden_layers):
            hl_neuron = neu.NeuronHidden(n_inputs, if_bias, momentum_coef, learn_coef, self.outputLayer, i + 1)
            self.hiddenLayer.append(hl_neuron)

        for i in range(n_outputs):
            ol_neuron = neu.NeuronOutput(n_hidden_layers, if_bias, momentum_coef, learn_coef, self.hiddenLayer)
            self.outputLayer.append(ol_neuron)

    # Function to keep the output of every output layer
    def f(self, x):
        out = []

        for i in range(len(self.outputLayer)):
            out.append(round(self.outputLayer[i].sigmoid_f(x), 2))
        return out

    def learn(self, x, y):

        f_out = open('output_weights.out', 'w')
        f_out.truncate(0)

        # calculate errors for every neuron in output layer
        for i in range(len(self.outputLayer)):
            self.outputLayer[i].sigma(x, y[i])

        # calculate errors for every neuron in hidden layer
        for obj in self.hiddenLayer:
            obj.sigma(x)

        # correct the weights
        for obj in self.outputLayer:
            ol_out = obj.correct(x)
            f_out.write(str(ol_out))

        for obj in self.hiddenLayer:
            hl_out = obj.correct(x)
            f_out.write(str(hl_out))

    # Step through one epoch
    def learn_epoch(self, x, y):
        for i in range(len(x)):
            self.learn(x[i], y[i])


def delta_Y(y, wyliczone):
    suma = 0

    for i in range(len(y)):
        for j in range(len(y[i])):
            suma += pow(wyliczone[i][j] - y[i][j], 2)

    return suma / (2 * len(y))


def delta_net(x, y, network):
    return delta_Y(y, [network.f(x[i]) for i in range(len(x))])


def outcome(x, network):
    return [network.f(x[i]) for i in range(len(x))]
