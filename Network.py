import Neuron as neu
import Parser as files


class Network:

    def __init__(self, n_inputs, n_hidden_layers, n_outputs, if_bias, learn_coef, momentum_coef, mode):

        # parameters:
        #   n_inputs - number of inputs
        #   n_hidden_layers - number of neurons in hidden layer
        #   n_outputs - number of desired outputs
        #   ifBias - decide if bias is added to calculation
        #   learn_coef - eta, learing coefficient
        #   momentum_coef - alfa, momentum coefficient
        self.mode = mode
        # Lists containing all neurons in the layers
        self.hiddenLayer = []
        self.outputLayer = []

        neu.NeuronHidden.sigmoid_f = neu.sigmoid
        neu.NeuronHidden.dF = neu.sigmoid_derivative
        neu.NeuronOutput.sigmoid_f = neu.sigmoid
        neu.NeuronOutput.dF = neu.sigmoid_derivative

        vector_weights = files.file_parser2('output_weights.out')
        vector_weights.reverse()

        # Initializing desired number of neurons in HL and OL, applying incrementing indexes to HL neurons
        j = n_hidden_layers
        for i in range(n_hidden_layers):
            hl_neuron = neu.NeuronHidden(n_inputs, if_bias, momentum_coef, learn_coef, mode, self.outputLayer, i + 1)

            if mode == "t":
                hl_neuron.setW(vector_weights[j-i-1])

            self.hiddenLayer.append(hl_neuron)

        j = n_hidden_layers
        for i in range(n_outputs):
            ol_neuron = neu.NeuronOutput(n_hidden_layers, if_bias, momentum_coef, learn_coef, mode, self.hiddenLayer)

            if mode == "t":
                ol_neuron.setW(vector_weights[j])
                j+=1

            self.outputLayer.append(ol_neuron)

    # Function to keep the output of every output layer
    def f(self, x):
        out = []

        for i in range(len(self.outputLayer)):
            out.append(round(self.outputLayer[i].sigmoid_f(x), 2))
        return out

    def f_test(self, x):
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

    def test(self, x, y):
        # calculate errors for every output layer
        for i in range(len(self.outputLayer)):
            self.outputLayer[i].sigma(x, y[i])

       # calculate errors for every hidden layer
        for obj in self.hiddenLayer:
            obj.sigma(x)

    def validate(self, x, y, x_test, y_test):

        f_out = open('output_weights.out', 'w')
        f_out.truncate(0)

        # calculate errors for every neuron in output layer
        for i in range(len(self.outputLayer)):
            self.outputLayer[i].sigma(x, y[i])
            self.outputLayer[i].sigma_test(x_test, y_test[i])

        # calculate errors for every neuron in hidden layer
        for obj in self.hiddenLayer:
            obj.sigma(x)
            obj.sigma_test(x_test)

        # correct the weights
        for obj in self.outputLayer:
            ol_out = obj.correct(x)
            f_out.write(str(ol_out))

        for obj in self.hiddenLayer:
            hl_out = obj.correct(x)
            f_out.write(str(hl_out))

    def test_epoch(self, x, y):
        for i in range(len(x)):
            self.test(x[i], y[i])

    def validate_epoch(self, x, y, x_test, y_test):
        for i in range(len(x)):
            self.validate(x[i], y[i], x_test[i], y_test[i])


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

# Changes for validation mode

def delta_net_test(x, y, network, x_test, y_test):
    result1 =  delta_Y(y, [network.f(x[i]) for i in range(len(x))])
    result2 = delta_Y(y_test, [network.f_test(x_test[i]) for i in range(len(x_test))])
    return result1, result2

def outcome_test(x, network, x_test):
    result1 = [network.f(x[i]) for i in range(len(x))]
    result2 = [network.f_test(x_test[i]) for i in range(len(x_test))]
    return result1, result2

# Changes for testing mode:
def outcome2(x, network):
    o = [network.f(x[i]) for i in range(len(x))]
    o.reverse()
    return o


def deltaY2(y, wyliczone):
    for i in range(len(wyliczone)):
        wyliczone[i].reverse()
    suma = 0

    for i in range(len(y)):
        for j in range(len(y[i])):
            suma += pow(wyliczone[i][j] - y[i][j], 2)

    return suma / (2 * len(y))


def deltaNet2(x, y, network):
    return deltaY2(y, [network.f(x[i]) for i in range(len(x))])
