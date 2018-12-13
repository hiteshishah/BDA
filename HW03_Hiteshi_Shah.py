"""
hw03_hiteshi_shah.py
author: Hiteshi Shah
description: To classify 1D data into unripe and ripe melons
"""

import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    bin_size = 0.05                   # constant bin size for the data

    # getting the data from the CSV file
    data = np.genfromtxt("CSCI720_2171_HW_03__v208_Classified_Data.csv", delimiter=",", skip_header=True)

    forces = data[:, 0]               # extracting the "force" column from the data
    deflections = data[:, 1]          # extracting the "deflection" column from the data
    classes = data[:, 2]              # extracting the "class" column from the data

    # calculating the flexibility threshold coefficient = deflection / force for each data point
    # and rounding the coefficients to 3 decimal points
    threshold_coefficients = (deflections / forces).round(decimals=3)

    # calculating the threshold for classifying the melons into unripe and ripe
    # the data points less than or equal to this threshold indicate unripe melons
    # and the data points more than it indicate ripe melons
    thresholds, misclass_rates, best_misclass_rate, best_threshold = classify_unripe_ripe(threshold_coefficients, bin_size, classes)

    # using the threshold calculated above with the training data, we classify the test data into unripe & ripe melons
    # and write the classification to a CSV file (class 0 indicates unripe melons and class 1 indicates ripe melons)
    classify_testing_data(best_threshold)

    # calculating the threshold for classifying the melons into ripe and rotting
    # the data points less than or equal to this threshold indicate ripe melons
    # and the data points more than it indicate rotting melons
    best_new_misclass_rate, best_new_threshold = classify_ripe_rotting(threshold_coefficients, bin_size, classes)

    # using the threshold calculated above with the training data, we classify the test data into ripe & rotting melons
    # and write the classification to a CSV file (class 1 indicates ripe melons and class 0 indicates rotting melons)
    classify_new_testing_data(best_new_threshold)

    # plotting the fraction of misclassified melons as function of the threshold
    plot(thresholds, misclass_rates, best_misclass_rate, best_threshold, best_new_misclass_rate, best_new_threshold)

def classify_unripe_ripe(threshold_coefficients, bin_size, classes):
    '''
    function to classify the given data points for melons into unripe and ripe melons
    :param threshold_coefficients: list of threshold coefficients of each data point
    :param bin_size: constant bin size for the data
    :param classes: list of classes assigned to each data point in the training data
    :return: returns the best threshold that minimizes the misclassification rate for the given data points
    '''

    best_misclass_rate = math.inf       # initializing the value for the best threshold to NaN

    thresholds = []                     # list of threshold values to be tested with
    misclass_rates = []                 # list of misclassification rates for each test threshold

    fars = []                           # list of false alarm rates for each test threshold
    tprs = []                           # list of true positive rates for each test threshold

    # calculating the best (minimum) misclassification rate and it's corresponding threshold, miss rate, false alarm
    # rate and true positive rate values using the "try them all" approach over a range of test thresholds
    for threshold in np.arange(bin_size, max(threshold_coefficients) + bin_size, bin_size):
        thresholds.append(threshold)
        n_fa = 0                        # number of false alarms
        n_misses = 0                    # number of misses
        n_hits = 0                      # number of hits
        n_cr = 0                        # number of correct rejections
        for index in range(0, len(threshold_coefficients)):
            if threshold_coefficients[index] <= threshold:
                # since "not unripe" is the target variable, if a melon classified as ripe is under this threshold,
                # then it's a miss, if it's classified as unripe, it's a correct rejection
                if classes[index] != 1:
                    n_misses += 1
                else:
                    n_cr += 1
            elif threshold_coefficients[index] > threshold:
                # since "not unripe" is the target variable, if a melon classified as unripe is over this threshold,
                # then it's a false alarm, if it's classified as ripe, it's a hit
                if classes[index] == 1:
                    n_fa += 1
                else:
                    n_hits += 1

        # calculating the fraction of misclassified melons
        n_wrong = round((n_fa + n_misses) / len(threshold_coefficients), 3)
        misclass_rates.append(n_wrong)
        # calculating the false negative rate
        if (n_misses + n_hits) != 0:
            miss_rate = round(n_misses / (n_misses + n_hits), 3)
        # calculating the true positive rate
        # and conditions for division by zero
        if (n_misses + n_hits) != 0:
            hit_rate = round(n_hits / (n_misses + n_hits), 3)
            tprs.append(hit_rate)
        else:
            tprs.append(1)
        # calculating the false alarm rate
        far = round(n_fa / (n_fa + n_cr), 3)
        fars.append(far)

        # saving the minimum misclassification rate and its associated values
        if n_wrong <= best_misclass_rate:
            best_misclass_rate = n_wrong
            best_threshold = threshold
            best_far = far
            best_miss_rate = miss_rate
            best_hit_rate = hit_rate
    print("FOR UNRIPE/RIPE")
    print("Best misclassification rate for unripe/ripe melons: " + str(best_misclass_rate))
    print("Best threshold that separates unripe melons from the ripe ones: " + str(best_threshold))
    print("Fraction of ripe melons missed (FNR): " + str(best_miss_rate))
    print("Fraction of unripe melons classified as ripe (FAR): " + str(best_far))

    # plotting the ROC curve for unripe/ripe melons
    plot_roc_curve(fars, tprs, best_far, best_hit_rate, "ROC Curve for unripe/ripe")

    # returning the threshold less than or equal to which are unripe melons, and more than which are ripe melons
    return thresholds, misclass_rates, best_misclass_rate, best_threshold

def classify_ripe_rotting(threshold_coefficients, bin_size, classes):
    '''
    function to classify the given data points for melons into ripe and rotting melons
    :param threshold_coefficients: list of threshold coefficients of each data point
    :param bin_size: constant bin size for the data
    :param classes: list of classes assigned to each data point in the training data
    :return: returns the best threshold that minimizes the misclassification rate for the given data points
    '''

    best_misclass_rate = math.inf  # initializing the value for the best threshold to NaN

    thresholds = []  # list of threshold values to be tested with
    misclass_rates = []  # list of misclassification rates for each test threshold

    fars = []  # list of false alarm rates for each test threshold
    tprs = []  # list of true positive rates for each test threshold

    # calculating the best (minimum) misclassification rate and it's corresponding threshold,
    # using the "try them all" approach over a range of test thresholds
    for threshold in np.arange(bin_size, max(threshold_coefficients) + bin_size, bin_size):
        thresholds.append(threshold)
        n_fa = 0            # number of false alarms
        n_misses = 0        # number of misses
        n_hits = 0          # number of hits
        n_cr = 0            # number of correct rejections
        total = 0           # total number of melons classified as ripe or rotting in the training data
        for index in range(0, len(threshold_coefficients)):
            # skip if the melon is classified as unripe
            if classes[index] == 1:
                continue
            else:
                total += 1
                if threshold_coefficients[index] <= threshold:
                    # since "not unripe" is the target variable, if a melon classified as ripe is under this threshold,
                    # then it's a hit, if it's classified as rotting, it's a false alarm
                    if classes[index] == 2:
                        n_hits += 1
                    else:
                        n_fa += 1
                elif threshold_coefficients[index] > threshold:
                    # since "not unripe" is the target variable, if a melon classified as rotting is over this threshold,
                    # then it's a correct rejection, if it's classified as ripe, it's a miss
                    if classes[index] == 3:
                        n_cr += 1
                    else:
                        n_misses += 1

        # calculating the fraction of misclassified melons
        n_wrong = round((n_fa + n_misses) / total, 3)
        misclass_rates.append(n_wrong)
        # calculating the true positive rate
        # and conditions for division by zero
        if (n_misses + n_hits) != 0:
            hit_rate = round(n_hits / (n_misses + n_hits), 3)
        else:
            hit_rate = 1
        tprs.append(hit_rate)
        # calculating the false alarm rate
        # and conditions for division by zero
        if (n_fa + n_cr) != 0:
            far = round(n_fa / (n_fa + n_cr), 3)
        else:
            far = 0
        fars.append(far)
        # saving the best (minimum) misclassification rate and its associated values
        if n_wrong <= best_misclass_rate:
            best_misclass_rate = n_wrong
            best_threshold = threshold
            best_far = far
            best_hit_rate = hit_rate
    print("\nFOR RIPE/ROTTING")
    print("Best misclassification rate for ripe/rotting melons: " + str(best_misclass_rate))
    print("Best threshold that separates ripe melons from the rotting ones: " + str(best_threshold))

    plot_roc_curve(fars, tprs, best_far, best_hit_rate, "ROC Curve for ripe/rotting")

    # returning the threshold less than or equal to which are ripe melons, and more than which are rotting melons
    return best_misclass_rate, best_threshold

def classify_testing_data(best_threshold):
    '''
    function to classify the test data using the threshold obtained from the training data
    :param best_threshold: the threshold for classifying the test data into unripe and ripe melons
    '''

    # getting the test data from the given file
    testing_data = np.genfromtxt("CSCI720_2171_HW_03__v208_MELONS_TO_CLASSIFY.csv", delimiter=",", skip_header=True)

    testing_forces = testing_data[:, 0]         # extracting the "force" column from the testing data
    testing_deflections = testing_data[:, 1]    # extracting the "deflection" column from the testing data

    # calculating the flexibility threshold coefficient = deflection / force for each data point
    testing_threshold_coefficients = (testing_deflections / testing_forces)

    testing_classes = []                        # list of classes for each data point in the test data
    results = []                                # the results of the classification to be written to a CSV file

    # assigning class 0 (unripe) to melons whose threshold is less than or equal to the given threshold
    # and class 1 (ripe) to melons whose threshold is more than the given threshold
    for t_index in range(0, len(testing_threshold_coefficients)):
        if testing_threshold_coefficients[t_index] <= best_threshold:
            testing_classes.append(0)
        else:
            testing_classes.append(1)
        results.append([testing_forces[t_index], testing_deflections[t_index], testing_classes[t_index]])

    # writing the results of the above classification to the CSV file below
    np.savetxt("HW03_Hiteshi_Shah_CLASSIFICATIONS.csv", results, fmt='%10.3f, %10.3f, %i', delimiter=",",
               header="Force (Newtons), Deflection (mm), Class", comments="")

def classify_new_testing_data(best_new_threshold):
    '''
    function to classify the test data using the threshold obtained from the training data
    :param best_threshold: the threshold for classifying the test data into ripe and rotting melons
    '''

    # getting the test data from the file containing the unripe and ripe classifications
    testing_data = np.genfromtxt("HW03_Hiteshi_Shah_CLASSIFICATIONS.csv", delimiter=",", skip_header=True)

    testing_forces = testing_data[:, 0]             # extracting the "force" column from the testing data
    testing_deflections = testing_data[:, 1]        # extracting the "deflection" column from the testing data

    classes = testing_data[:, 2]                    # extracting the "class" column from the testing data

    # calculating the flexibility threshold coefficient = deflection / force for each data point
    testing_threshold_coefficients = (testing_deflections / testing_forces)

    forces = []                                     # list of forces that don't include the unripe melons
    deflections = []                                # list of deflections that don't include the unripe melons
    results = []                                    # the results of the classification to be written to a CSV file

    # assigning class 1 (ripe) to melons whose threshold is less than or equal to the given threshold
    # and class 0 (rotting) to melons whose threshold is more than the given threshold
    for t_index in range(0, len(testing_threshold_coefficients)):
        # skip if the melon is already classified as unripe
        if classes[t_index] == 0:
            continue
        else:
            forces.append(testing_forces[t_index])
            deflections.append(testing_deflections[t_index])
            if testing_threshold_coefficients[t_index] <= best_new_threshold:
                class_value = 1
            else:
                class_value = 0
            results.append([testing_forces[t_index], testing_deflections[t_index], class_value])

    # writing the results of the above classification to the CSV file below
    np.savetxt("HW03_Hiteshi_Shah_CLASSIFICATIONS_INTO_RIPE_AND_ROTTING.csv", results, fmt='%10.3f, %10.3f, %i',
               delimiter=",", header="Force (Newtons), Deflection (mm), Class", comments="")


def plot(thresholds, misclass_rates, best_misclass_rate, best_threshold, best_new_misclass_rate, best_new_threshold):
    '''
    function to plot the fraction of misclassified melons as a function of the threshold
    :param threshold_coefficients: list of threshold coefficients of all the data points
    :param bin_size: the bin size for the data
    :param best_misclass_rate: the best (minimum) rate of misclassification for unripe/ripe melons
    :param best_threshold: the threshold value for splitting the unripe and ripe melons
    :param best_new_ misclass_rate: the best (minimum) rate of misclassification for ripe/rotting melons
    :param best_new_threshold: the threshold value for splitting the ripe and rotting melons
    '''

    plt.plot(thresholds, misclass_rates)                    # thresholds v/s their corresponding misclassification rates
    plt.axis([0, thresholds[-1], 0, max(misclass_rates)])   # axes scales
    # vertical red line from 0 to best_misclass_rate
    plt.plot([best_threshold, best_threshold], [0, best_misclass_rate], "r", label="unripe/ripe")
    # horizontal red line from 0 to best_threshold
    plt.plot([best_threshold, 0], [best_misclass_rate, best_misclass_rate], "r")
    # vertical yellow line from 0 to best_new_misclass_rate
    plt.plot([best_new_threshold, best_new_threshold], [0, best_new_misclass_rate], "y", label="ripe/rotting")
    # horizontal yellow line from 0 to best_new_threshold
    plt.plot([best_new_threshold, 0], [best_new_misclass_rate, best_new_misclass_rate], "y")
    plt.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0.)            # legend for the vertical lines
    plt.xlabel("thresholds")                                              # label for the x-axis
    plt.ylabel("misclassification rates")                                 # label for the y-axis
    plt.title("misclassification rate as a function of threshold")        # title of the graph
    plt.show()                                                            # displaying the graph

def plot_roc_curve(fars, tprs, best_far, best_hit_rate, title):
    '''
    function to plot the ROC curve
    :param fars: list of false alarm rates
    :param tprs: list of true positive rates
    :param best_far: best false alarm rate associated with the minimum misclassification rate
    :param best_hit_rate: best true positive rate associated with the minmum misclassification rate
    :param title: title for the graph to be plotted
    '''
    plt.plot(fars, tprs)                        # false alarm rates v/s true positive rates
    plt.axis([0, max(fars), 0, max(tprs)])      # axes scales
    plt.plot(best_far, best_hit_rate, 'ro')     # marker indicating the best threshold on the ROC curve
    plt.xlabel("False Alarm Rates")             # label for the x-axis
    plt.ylabel("True Postive Rates")            # label for the y-axis
    plt.title(title)                            # title of the graph
    plt.show()                                  # displaying the graph

main()