import re


def file_parser1(filename):
    vector_X = []

    with open(filename) as plik:
        tab = plik.read()

    tab = re.split('\n', tab)

    while '' in tab:
        tab.remove('')

    for line in tab:
        rowTxt = re.split(',', line)
        row = [float(x) for x in rowTxt]
        vector_X.append(row)
    return vector_X


def file_parser2(filename):

    vector_X = []

    with open(filename) as plik:
        tab = plik.read()
        tab = tab.replace("]", "")

    tab = re.split('\[', tab)

    while '' in tab:
        tab.remove('')

    for line in tab:
        rowTxt = re.split(',', line)
        row = [float(x) for x in rowTxt]

        vector_X.append(row)

    return vector_X

def whitch_columns(original_array, indexes):
    if indexes[3] == "0":
        for row in original_array:
            del row[3]
    if indexes[2] == "0":
        for row in original_array:
            del row[2]
    if indexes[1] == "0":
        for row in original_array:
            del row[1]
    if indexes[0] == "0":
        for row in original_array:
            del row[0]
    return original_array


def take_user_input():
    # Take user input
    which_columns = []
    sepal_length = input("Consider sepal length? y/n: ")
    if sepal_length == "y":
        which_columns.append("1")
    else:
        which_columns.append("0")

    sepal_width = input("Consider sepal width? y/n: ")
    if sepal_width == "y":
        which_columns.append("1")
    else:
        which_columns.append("0")

    petal_length = input("Consider petal length? y/n: ")
    if petal_length == "y":
        which_columns.append("1")
    else:
        which_columns.append("0")

    petal_width = input("Consider petal_width? y/n: ")
    if petal_width == "y":
        which_columns.append("1")
    else:
        which_columns.append("0")

    #define number of neurons in input layer
    input_layer_neurons = 0
    for i in which_columns:
        if i != "0":
            input_layer_neurons += 1

    # define number of neurons in hidden layer
    hidden_layer_neurons = int(input("Number of neurons in hidden layer: "))
    output_layer_neurons = 3
    # define if bias
    if_bias = input("Consider bias? y/n: ")
    if if_bias == "y":
        bias = 1
    else:
        bias = 0

    # lern coefficient
    eta = float(input("Learn coefficient value: "))
    momentum = float(input("Momentum coefficient value: "))

    return which_columns, input_layer_neurons, hidden_layer_neurons, output_layer_neurons, bias, eta, momentum
