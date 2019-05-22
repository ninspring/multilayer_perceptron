import Network as net
import Parser as files
import matplotlib.pyplot as plt
import numpy as np


# Name of the input and output file (in this case it's the same)
vector_x, vector_y = files.file_parser1('pattern1.txt')

# Name of the file containing weights for testing mode
vector_weights = files.file_parser2('output_weights.out')


# Erase previous info from the input file
f_in = open('input_weights.out', 'w')
f_in.truncate(0)
f_in.close()


iterations = 5000
precision = 0.01

# Define network
mode = "t"                  # "l" for learning mode, "t" for testing
option = "i"                # "i" for number of iterations, "p" for precision
input_layer_neurons = 4
hidden_layer_neurons = 2
output_layer_neurons = 4
bias = 1
eta = 0.6
momentum = 0

print("------------------------------")
print("Ustawienia sieci:")
print("------------------------------")
print("liczba neuronów w warstwie wejściowej: ", input_layer_neurons)
print("liczba neuronów w warstwie ukrytej: ", hidden_layer_neurons)
print("liczba neuronów w warstwie wyjściowej: ", output_layer_neurons)

if bias == 0:
    print("bias: NIE")
else:
    print("bias: TAK")
print("wspólczynnik nauki: ", eta)
print("wspólczynnik momentum: ", momentum, "\n")

if mode == "l":
    # generate plots

    network = net.Network(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, bias, eta, momentum, mode)

    vec_delta = []
    vec_outcome = []

    print("------------------------------")
    print("Błędy wypisywane co 200 epok:")
    print("------------------------------")

    if option == "i":
        for i in range(iterations):
            error = net.deltaNet(vector_x, vector_y, network)
            vec_delta.append(error)
            vec_outcome.append(net.outcome(vector_x, network))
            network.learn_epoch(vector_x, vector_y)
            if i%200 == 0:
                print("Błąd w epoce ", i, "wynosi: ", round(error, 4))

    if option == "p":
        error = 100
        i = 0
        while error > precision:
            i += 1
            error = net.deltaNet(vector_x, vector_y, network)
            vec_delta.append(error)
            vec_outcome.append(net.outcome(vector_x, network))
            network.learn_epoch(vector_x, vector_y)
            if i%200 == 0:
                print("Błąd w epoce ", i, "wynosi: ", round(error, 4))


    print("------------------------------")
    print("Błąd obliczony w ostatniej epoce:", vec_delta[-1])
    print("Wzorzec: ", vector_y)
    print("Wynik:   ", vec_outcome[-1])

    plt.figure(0)
    line1, = plt.plot(vec_delta, 'r', label='1')
    plt.legend(handles=[line1], title="Error per iteration")
    plt.title('Neurons in hidden layer: {}'.format(hidden_layer_neurons))
    plt.xlabel("iterations")
    plt.ylabel("error")
    plt.grid()
    plt.show()


if mode == "t":

    network = net.Network(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, bias, eta, momentum, mode)

    vec_delta = []
    vec_outcome = []

    for i in range(len(vector_x)):
        error = net.deltaNet2(vector_x, vector_y, network)
        vec_delta.append(error)
        vec_outcome.append(net.outcome2(vector_x, network))
        network.test_epoch(vector_x, vector_y)


    print("------------------------------")
    print("Błąd obliczony w ostatniej epoce:", vec_delta[-1])
    print("Wzorzec: ", vector_y)
    print("Wynik:   ", vec_outcome[0])


