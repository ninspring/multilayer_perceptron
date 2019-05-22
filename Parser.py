import re


def file_parser1(filename):
    vector_X = []

    with open(filename) as plik:
        tab = plik.read()

    tab = re.split('\n', tab)

    while '' in tab:
        tab.remove('')

    for line in tab:
        rowTxt = re.split(' ', line)
        row = [float(x) for x in rowTxt]

        vector_X.append(row)
    return vector_X, vector_X

def file_parser2(filename):
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