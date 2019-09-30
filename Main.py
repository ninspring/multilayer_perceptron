import Network as net
import Parser as files
import matplotlib.pyplot as plt
import Classification
import Graphical_Interface

# Name of the input and output file (in this case it's the same)
vector_X = files.file_parser('irisX_learn.txt')
vector_y = files.file_parser('irisY_learn.txt')

#Take user input to define which features take into consideration
result = files.take_user_input()
vector_x = files.which_columns(vector_X, result[0])

# Erase previous info from the input file
f_in = open('input_weights.out', 'w')
f_in.truncate(0)
f_in.close()

iterations = int(Graphical_Interface.values['_its_'])
precision = 0.001

# Define network
input_layer_neurons = result[1]
hidden_layer_neurons = result[2]
output_layer_neurons = result[3]
bias = result[4]
eta = result[5]
momentum = result[6]

print("------------------------------")
print("Network settings:")
print("------------------------------")
print("Neurons in input layer: ", input_layer_neurons)
print("Neurons in hidden layer: ", hidden_layer_neurons)
print("Neurons in output layer: ", output_layer_neurons)
Graphical_Interface.window.Refresh()

if bias == 0:
    print("bias: NO")
else:
    print("bias: YES")
print("learn coefficient value: ", eta)
print("momentum coefficient value: ", momentum, "\n")

# generate plots
network = net.Network(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, bias, eta, momentum)

vec_delta = []
vec_outcome = []

print("------------------------------")
print("Error in every 200-th epoch:")
print("------------------------------")
Graphical_Interface.window.Refresh()

epoka = -1
error = 100
for i in range(iterations):
    if error > precision:
        error = net.delta_net(vector_x, vector_y, network)
        vec_delta.append(error)
        vec_outcome.append(net.outcome(vector_x, network))
        network.learn_epoch(vector_x, vector_y)
        if i % 200 == 0:
            print("Error in epoch no. ", i, "equals: ", round(error, 4))
            epoka = i
            Graphical_Interface.window.Refresh()
        i += 1



print("------------------------------")
print("Error in last epoch equals:", vec_delta[-1])
print("Number of iterations:", epoka)
Graphical_Interface.window.Refresh()

target = Classification.name_species(vector_y, vec_delta[-1])
predicted = Classification.name_species(vec_outcome[-1], vec_delta[-1])
print("------------------------------")
print("Confusion matrix: ")
print(Classification.skliearnmatrix(target, predicted))
Graphical_Interface.window.Refresh()

plt.figure(0)
line1, = plt.plot(vec_delta, 'r', label='learning set')
plt.legend(handles=[line1], title="Error per iteration")
plt.title('Neurons in hidden layer: {}'.format(hidden_layer_neurons))
plt.xlabel("iterations")
plt.ylabel("error")
plt.ylim(bottom=0.0)
plt.xlim(left=0.0)
plt.grid()
plt.show()

Classification.visualize()