from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def name_species(predict, error):
    class_result = []
    if error < 0.1:
        error = 0.1

    for i in range(len(predict)):
        if (1-abs(predict[i][0])) <= error and (0-abs(predict[i][1])) <= error and (0-abs(predict[i][2])) <= error:
            class_result.append("setosa")
        elif (0 - abs(predict[i][0])) <= error and (1 - abs(predict[i][1])) <= error and (0 - abs(predict[i][2])) <= error:
            class_result.append("versicolor")
        elif (0 - abs(predict[i][0])) <= error and (0 - abs(predict[i][1])) <= error and (1 - abs(predict[i][2])) <= error:
            class_result.append("viriginica")
        else:
            class_result.append("not classified")
    return class_result


def skliearnmatrix(target, predict):
    return confusion_matrix(target, predict, labels=["setosa", "versicolor", "viriginica"])


def visualize():
    sns.set(style="ticks", color_codes=True)
    data = pd.read_csv("iris.csv")
    g = sns.pairplot(data, hue="variety")
    plt.show()
