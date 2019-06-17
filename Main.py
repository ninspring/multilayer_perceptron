import Network as net
import Parser as files
import matplotlib.pyplot as plt
import Classification

'''
#########################TESTING PURPOSES#####
vector_x = files.file_parser1('irisXa.txt')
vector_y = files.file_parser1('irisYa.txt')
vector_test_x = files.file_parser1('irisXb.txt')
vector_test_y = files.file_parser1('irisYb.txt')
mode = "v"                  # "l" for learning mode, "t" for testing, "v" for validation
input_layer_neurons = 4
hidden_layer_neurons = 8
output_layer_neurons = 3
bias = 1
eta = 0.6
momentum = 0
'''
# Name of the input and output file (in this case it's the same)
vector_X = files.file_parser1('irisX_learn.txt')
vector_y = files.file_parser1('irisY_learn.txt')
vector_test_X = files.file_parser1('irisX_test.txt')
vector_test_y = files.file_parser1('irisY_test.txt')

#Take user input to define which fetures take into consideration
result = files.take_user_input()
vector_x = files.whitch_columns(vector_X, result[0])
vector_test_x = files.whitch_columns(vector_test_X, result[0])

# Name of the file containing weights for testing mode
vector_weights = files.file_parser2('output_weights.out')


# Erase previous info from the input file
f_in = open('input_weights.out', 'w')
f_in.truncate(0)
f_in.close()


iterations = 200
precision = 0.001

# Define network
mode = "v"                  # "l" for learning mode, "t" for testing, "v" for validation
input_layer_neurons = result[1]
hidden_layer_neurons = result[2]
output_layer_neurons = result[3]
bias = result[4]
eta = result[5]
momentum = result[6]

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

Classification.visualize()


if mode == "l":
    # generate plots

    network = net.Network(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, bias, eta, momentum, mode)

    vec_delta = []
    vec_outcome = []

    print("------------------------------")
    print("Błędy wypisywane co 200 epok:")
    print("------------------------------")

    epoka = 0
    error = 100
    for i in range(iterations):
        if error > precision:
            i += 1
            error = net.delta_net(vector_x, vector_y, network)
            vec_delta.append(error)
            vec_outcome.append(net.outcome(vector_x, network))
            network.learn_epoch(vector_x, vector_y)
            if i % 200 == 0:
                print("Błąd w epoce ", i, "wynosi: ", round(error, 4))
                epoka = i


    print("------------------------------")
    print("Błąd obliczony w ostatniej epoce:", vec_delta[-1])
    print("Ilość iteracji:", epoka)
    print("Wzorzec: ", vector_y)
    print("Wynik:   ", vec_outcome[-1])

    target = Classification.name_species(vector_y, vec_delta[-1])
    predicted = Classification.name_species(vec_outcome[-1], vec_delta[-1])
    print(target)
    print(predicted)
    print(Classification.skliearnmatrix(target, predicted))


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
    print("Błąd obliczony:", vec_delta[-1])
    print("Wzorzec: ", vector_y)
    print("Wynik:   ", vec_outcome[0])

if mode == "v":
    # generate plots

    network = net.Network(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, bias, eta, momentum, mode)

    vec_delta = []
    vec_outcome = []
    vec_delta_test = []
    vec_outcome_test = []

    print("------------------------------")
    print("Błędy wypisywane co 200 epok:")
    print("------------------------------")

    epoka = 0
    error = (100, 100)

    for i in range(iterations):
        if error[0] > precision:
            i += 1
            error = net.delta_net_test(vector_x, vector_y, network, vector_test_x, vector_test_y)
            outcome = net.outcome_test(vector_x, network, vector_test_x)

            vec_delta.append(error[0])
            vec_outcome.append(outcome[0])

            vec_delta_test.append(error[1])
            vec_outcome_test.append(outcome[1])

            network.validate_epoch(vector_x, vector_y, vector_test_x, vector_test_y)
            if i % 200 == 0:
                epoka = i
                print("Błąd dla zbioru uczącego w epoce ", i, "wynosi: ", round(error[0], 4))
                print("Błąd dla zbioru testującego w epoce ", i, "wynosi: ", round(error[1], 4))


    print("------------------------------")
    print("Błąd dla zbioru uczącego obliczony w ostatniej epoce:", vec_delta[-1])
    print("Błąd dla zbioru testującego obliczony w ostatniej epoce:", vec_delta_test[-1])
    print("Ilość iteracji:", epoka)
    print("------------------------------")
    print("Zbiór uczący: ")
    print("Wzorzec: ", vector_y)
    print("Wynik:   ", vec_outcome[-1])
    print("------------------------------")
    print("Zbiór testowy: ")
    print("Wzorzec: ", vector_test_y)
    print("Wynik:   ", vec_outcome_test[-1])

    target = Classification.name_species(vector_y, vec_delta[-1])
    predicted = Classification.name_species(vec_outcome[-1], vec_delta[-1])
    print("------------------------------")
    print("Tablica pomyłek dla zbioru uczącego: ")
    print(Classification.skliearnmatrix(target, predicted))

    target_test = Classification.name_species(vector_test_y, vec_delta_test[-1])
    predicted_test = Classification.name_species(vec_outcome_test[-1], vec_delta_test[-1])
    print("------------------------------")
    print("Tablica pomyłek dla zbioru testowego: ")
    print(Classification.skliearnmatrix(target_test, predicted_test))

    #vec_delta = vec_delta[::10]
    #vec_delta_test = vec_delta_test[::10]

    plt.figure(0)
    line1, = plt.plot(vec_delta, 'r', label='zbior uczacy')
    line2, = plt.plot(vec_delta_test, 'b', label='zbior testowy')
    plt.legend(handles=[line1, line2], title="Error per iteration")
    plt.title('Neurons in hidden layer: {}'.format(hidden_layer_neurons))
    plt.xlabel("iterations")
    plt.ylabel("error")
    plt.ylim(bottom=0.0)
    plt.xlim(left=0.0)
    plt.grid()
    plt.show()
