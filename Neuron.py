import numpy as np
import random


class Neuron:

    def __init__(self, n_inputs, if_bias, momentum_coef, learn_coef, mode):
        self.n_inputs = n_inputs            # number of inputs in a neuron
        self.bias = if_bias
        self.momentum_coef = momentum_coef  # alfa
        self.learn_coef = learn_coef        # eta

        # w - weights matrix
        if mode == "l" or "v":
            self.w = [random.uniform(-1, 1) for x in range(n_inputs + 1)]

            with open('input_weights.out', 'a') as f_handle:
                f_handle.write(str(self.w))
                f_handle.close()

        # sigma - error, predefined with 0
        self.current_sigma = 0
        self.last_sigma = [0 for x in range(n_inputs+1)]
        # error for testing data, predefined with 0
        self.current_sigma_test = 0
        self.last_sigma_test = [0 for x in range(n_inputs+1)]

    # take calculated sigma (already multiplied by sigmoid derivative and eta) and multiply by momentum coef
    def correct(self, x):

        if self.bias:
            delta_w = self.current_sigma + self.momentum_coef * self.last_sigma[0]
            self.w[0] -= delta_w
            self.last_sigma[0] = delta_w

        for i in range(self.n_inputs):
            delta_w = self.current_sigma * self.correct_coef(x, i) + self.last_sigma[i + 1] * self.momentum_coef
            self.w[i + 1] -= delta_w
            self.last_sigma[i + 1] = delta_w

        return self.w

    def setW(self, w):
        self.w = w


# Class Neuron from the Hidden Layer, inherits from class Neuron
class NeuronHidden(Neuron):

    def __init__(self, n_inputs, if_bias, momentum_coef, learn_coef, mode, output_layer, index):
        Neuron.__init__(self, n_inputs, if_bias, momentum_coef, learn_coef, mode)

        self.output_layer = output_layer    # number of output layers = the number of inputs for the neuron in HL in BP
        self.index = index

    # function to calculate the sum in the Neuron
    def sumator(self, X):
        suma = 0

        if self.bias:
            suma += self.w[0]

        for i in range(len(X)):
            suma += self.w[i + 1] * X[i]
        return suma

    # function to calculate the sigma error
    #       sigma(n) = sum nextLayerSigma * weight(n)
    def sigma(self, x):
        dF = self.dF(x)
        self.current_sigma = 0

        # for every Output Layer, take it's calculated sigma error and multiply it by the weight of the input
        for i in range(len(self.output_layer)):
            self.current_sigma += self.output_layer[i].current_sigma * self.output_layer[i].w[self.index]

        # we take the calculated sigma for the current Neuron we're in and multiply it by the activation function
            # derivative and by eta (learn coef). Essential for weights modification
        self.current_sigma = (self.current_sigma * dF * self.learn_coef)

    def sigma_test(self, x):
        dF = self.dF(x)
        self.current_sigma_test = 0

        # for every Output Layer, take it's calculated sigma error and multiply it by the weight of the input
        for i in range(len(self.output_layer)):
            self.current_sigma_test += self.output_layer[i].current_sigma_test * self.output_layer[i].w[self.index]

        # we take the calculated sigma for the current Neuron we're in and multiply it by the activation function
            # derivative and by eta (learn coef). Essential for weights modification
        self.current_sigma_test = (self.current_sigma_test * dF * self.learn_coef)

    def correct_coef(self, x, i):
        return x[i]


# Class Neuron from the Output Layer, inherits from class Neuron

class NeuronOutput(Neuron):

    def __init__(self, n_inputs, if_bias, momentum_coef, learn_coef, mode, hidden_layer):
        Neuron.__init__(self, n_inputs, if_bias, momentum_coef, learn_coef, mode)

        self.hidden_layer = hidden_layer  # list containing hidden layers

    # function to calculate the sum in the Neuron
    def sumator(self, X):
        suma = 0

        if self.bias:
            suma += self.w[0]
        for i in range(len(self.hidden_layer)):
            suma += self.w[i + 1] * self.hidden_layer[i].sigmoid_f(X)   # multiply weights and hidden layer output

        return suma

    # function to calculate sigma error in the output layer by comparing it to the desired output y
    def sigma(self, x, y):
        f = self.sigmoid_f(x)
        dF = self.dF(x)

        self.current_sigma = ((f - y) * dF) * self.learn_coef

    def sigma_test(self, x, y):
        f = self.sigmoid_f(x)
        dF = self.dF(x)

        self.current_sigma_test = ((f - y) * dF) * self.learn_coef

    def correct_coef(self, x, i):
        return self.hidden_layer[i].sigmoid_f(x)


# END OF NEURON CLASSES


def sigmoid(self, x):
    suma = self.sumator(x)
    return 1 / (1 + np.exp(-suma))


def sigmoid_derivative(self, x):
    y = self.sigmoid_f(x)
    return y * (1 - y)
