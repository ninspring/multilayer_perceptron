import re
import Graphical_Interface as gui


def file_parser(filename):
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


def which_columns(original_array, indexes):
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
    if gui.values['_sep_len_'] == True:
        which_columns.append("1")
    else:
        which_columns.append("0")

    if gui.values['_sep_wid_'] == True:
        which_columns.append("1")
    else:
        which_columns.append("0")

    if gui.values['_pet_len_'] == True:
        which_columns.append("1")
    else:
        which_columns.append("0")

    if gui.values['_pet_wid_'] == True:
        which_columns.append("1")
    else:
        which_columns.append("0")

    #define number of neurons in input layer
    input_layer_neurons = 0
    for i in which_columns:
        if i != "0":
            input_layer_neurons += 1

    # define number of neurons in hidden layer
    hidden_layer_neurons = int(gui.values['_hidden_'])
    output_layer_neurons = 3
    # define if bias
    if gui.values['_bias_'] == True:
        bias = 1
    else:
        bias = 0

    # lern coefficient
    eta = float(gui.values['_eta_'])
    momentum = float(gui.values['_alfa_'])

    return which_columns, input_layer_neurons, hidden_layer_neurons, output_layer_neurons, bias, eta, momentum
