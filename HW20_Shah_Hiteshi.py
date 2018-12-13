"""
hw20_shah_hiteshi.py
author: Hiteshi Shah
date: 12/6/2017
description: To perform sequential analysis using bigrams and trigrams
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    # getting the training data from the CSV file
    training_data = np.genfromtxt("FLD_HW__TRAINING_v22.csv", delimiter=",", skip_header=True)

    # plotting the data
    plot(training_data)

    training_data = training_data.T

    # separating the target variable from the rest of the attributes in the training data
    target = training_data[2]
    training_data = training_data[:2].T

    # training the LDA classifier with the training data
    clf = LinearDiscriminantAnalysis()
    clf = clf.fit(training_data, target)
    X_clf = clf.transform(training_data)

    # getting the testing data from the CSV file
    data = np.genfromtxt ("FLD_HW__TESTING_v22.csv",  dtype="str", delimiter="\r")
    testing_data = []
    for i in range(1, len(data)):
        numbers = data[i].split(",")
        testing_data.append([float(numbers[0]), float(numbers[1])])

    # performing classification on the testing data using the model obtained from the training data
    testing_data = np.array(testing_data)
    Z_clf = clf.transform(testing_data)
    predictions = clf.predict(testing_data)

    # saving the results to a CSV file
    np.savetxt("HW20_Hiteshi_Shah.csv", predictions, fmt="%s", delimiter=",", comments="")

def plot(training_data):
    '''
    function to plot the training data
    :param training_data: the given training data
    '''

    # separating the training data into 'age' and 'height' features
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for point in training_data:
        if point[2] == 1:
            x1.append(point[0])
            y1.append(point[1])
        if point[2] == 0:
            x2.append(point[0])
            y2.append(point[1])


    # plotting the separated data on a scatterplot
    plt.scatter(x1, y1, color='r')
    plt.scatter(x2, y2, color='b')
    plt.xlabel("Age")
    plt.ylabel("Height")
    plt.show()

main()