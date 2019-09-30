import PySimpleGUI as sg

layout = [[sg.Text('Multi Layer Perceptron', size=(30, 1), font=("Helvetica", 25), text_color='black')],
    [sg.Text('Neurons in hidden layer'), sg.Input('5', key='_hidden_')],
    [sg.Text('Learn coefficient value'), sg.Input('0.1', key='_eta_')],
    [sg.Text('Momentum coefficient value'), sg.Input('0.2', key='_alfa_')],
    [sg.Checkbox('Consider bias?', default=True, key='_bias_')],
    [sg.Checkbox('Consider sepal length?', default=True, key='_sep_len_')],
    [sg.Checkbox('Consider sepal width?', default=True, key='_sep_wid_')],
    [sg.Checkbox('Consider petal length?', default=True, key='_pet_len_')],
    [sg.Checkbox('Consider petal width?', default=True, key='_pet_wid_')],
    [sg.Spin([i for i in range(1,10000)], initial_value=500, key='_its_'), sg.Text('Number of iterations')],
    [sg.Output (size=(50, 10),
                key='_output_',
                visible=True)],
    [sg.Submit(), sg.Cancel()]]

window = sg.Window('MLP application').Layout(layout).Finalize()

event, values = window.Read()
