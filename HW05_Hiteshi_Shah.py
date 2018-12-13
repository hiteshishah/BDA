"""
hw05_hiteshi_shah.py
author: Hiteshi Shah
description: To find the five highest correlation coefficients and their variables
"""

import numpy as np

def pcc(X, Y):
    '''
    function to calculate the Pearson's correlation coefficient between variables X and Y
    :param X: list of values of the first variable
    :param Y: list of values of the second variable
    :return: the Pearson's correlation coefficient between variables X and Y
    '''
    X -= X.mean(0)          # calculates the mean of all the values in X
    Y -= Y.mean(0)          # calculates the mean of all the values in Y
    if X.std(0) == 0:
        X = 0
    else:
        X /= X.std(0)       # divides the previously calculated mean by the standard deviation of the values in X
    if Y.std(0) == 0:
        Y = 0
    else:
        Y /= Y.std(0)       # divides the previously calculated mean by the standard deviation of the values in Y

    return np.mean(X * Y)   # returns the Pearson's correlation coefficient

def main():
    # getting the data from the CSV file
    data = np.genfromtxt("Data_Population_Survey_as_Binary_v700.csv", delimiter=",", dtype="unicode")

    # separating the headers from the rest of the data
    columns = data[0]
    data = data[1:].T.astype(np.float)

    n = len(data)

    # intializing the matrix that consists of the correlation coefficients of all the variables
    cormatrix = np.zeros(shape=(n, n))

    # calculating the pairwise correlation coefficients of all the variables,
    # keeping the diagonals and the lower part of the matrix as zeros
    for i in range(0, n):
        for j in range(0, n):
            if i < j:
                cormatrix[i, j] = round(pcc(data[i], data[j]), 3)

    print("Value\t\tAttribute One\t\t\t\tAttribute Two")

    # finding out and displaying the top 5 correlation coefficients and their variables
    for _ in range(0, 5):
        maxvalue = np.amax(cormatrix)
        indices = np.where(cormatrix == maxvalue)

        print(str(maxvalue) + "\t\t" + columns[indices[0][0]] + "\t\t\t\t" + columns[indices[1][0]])

        cormatrix[indices[0][0], indices[1][0]] = 0

main()