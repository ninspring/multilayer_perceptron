import numpy as np
from sklearn.metrics import confusion_matrix

import math


def name_species(predict, error):
    class_result = []

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
