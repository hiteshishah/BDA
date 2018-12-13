"""
hw02_shah_hiteshi.py
author: Hiteshi Shah
description: To demonstrate the gradient descent algorithm using Otsu's method
"""

import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    bin_size = 0.05                   # constant bin size for the data

    # getting the data from the CSV file
    data = np.genfromtxt("CSCI720_2171_HW_02_Otsu__v208_Unclassified_Data.csv", delimiter=",", skip_header=True)

    forces = data[:, 0]               # extracting the "force" column from the data
    deflections = data[:, 1]          # extracting the "deflection" column from the data

    # calculating the flexibility threshold coefficient = deflection / force for each data point
    # and rounding the coefficients to 3 decimal points
    threshold_coefficients = (deflections / forces).round(decimals=3)

    # sorting the the threshold coefficients in ascending order
    threshold_coefficients.sort()

    # calculating the best mixed variance and it's corresponding threshold
    # the data points less than or equal to this threshold indicate unripe melons
    # and the data points more than it indicate ripe melons
    best_mixed_variance, best_threshold = using_gradient_descent(threshold_coefficients, bin_size)

    print("Unripe/ripe")
    print("Best mixed variance: " + str(best_mixed_variance))
    print("Best threshold: " + str(round(best_threshold, 3)))

    # trimming the list of threshold coefficients to get rid of the data points that belong to unripe melons
    # and calculating the threshold for splitting the data points to ripe and rotting melons
    new_threshold_coefficients = threshold_coefficients[threshold_coefficients > best_threshold]

    # calculating the best mixed variance and it's corresponding threshold
    # the data points less than or equal to this threshold indicate ripe melons
    # and the data points more than it indicate rotting melons
    best_new_mixed_variance, best_new_threshold = using_gradient_descent(new_threshold_coefficients, bin_size)

    print("\nRipe/rotting")
    print("Best mixed variance: " + str(best_new_mixed_variance))
    print("Best threshold: " + str(round(best_new_threshold, 3)))

    # plotting the graph of mixed variance as a function of the threshold values
    plot(threshold_coefficients, bin_size, best_threshold, best_new_threshold)


def using_gradient_descent(threshold_coefficients, bin_size):
    """
    function that implements the gradient descent algorithm to minimize the mixed variance found by using Otsu's method
    :param threshold_coefficients: list of threshold coefficients of each data point
    :param bin_size: constant bin size for the data
    :return: returns the best (min) mixed variance and it's corresponding threshold value
    """
    best_threshold = math.nan           # initializing the value for the best threshold to NaN

    step_size = 0.02                    # the step size being using for the gradient descent
    x = threshold_coefficients[18]      # initial input threshold value

    # calculating the initial mixed variance at x for comparison
    best_mixed_variance = calculate_mixed_variance(threshold_coefficients, x)

    # calculating the best (min) mixed variance using gradient descent by changing the value of step size
    # the algorithm stops when the step size equals the bin size
    while abs(round(step_size, 2)) != bin_size:
        # incrementing or decrementing the value of the threshold depending on the result of the function
        x += step_size

        # calculating the mixed variance at the new x
        mixed_variance = calculate_mixed_variance(threshold_coefficients, x)

        # if the newly calculated mixed variance is less than the previous best mixed variance,
        # i.e., the function moves towards the minimum mixed variance,
        # then assign the best mixed variance and best threshold to the current mixed variance and it's threshold
        # and expand the step size
        if mixed_variance <= best_mixed_variance:
            best_mixed_variance = mixed_variance
            best_threshold = x
            step_size = step_size + 0.06 * abs(step_size)

        # if the newly calculated mixed variance is more than the previous best mixed variance,
        # i.e., the function moves away from the minimum mixed variance,
        # then shrink the step size
        elif mixed_variance > best_mixed_variance:
            step_size = step_size - 2.5 * abs(step_size)

    return best_mixed_variance, best_threshold


def calculate_mixed_variance(threshold_coefficients, threshold):
    """
    function for calculating the mixed variance using Otsu's method,
    given the list of threshold coefficients and the threshold for the split
    :param threshold_coefficients: list of threshold coefficients
    :param threshold: threshold for splitting the data
    :return: the mixed variance rounded to 3 decimal points
    """
    total = threshold_coefficients.sum()  # sum of all the threshold coefficients

    wt_under = 0          # initializing the weight of the fraction of points less than or equal to the given threshold
    wt_over = 0           # initializing the weight of the fraction of points more than the given threshold
    no_under = 0          # initializing the number of points that are less than or equal to the given threshold
    no_over = 0           # initializing the number of points that are more than the given threshold

    # calculating the sum of threshold coefficients less than or equal to the given threshold,
    # the sum of threshold coefficients more to the given threshold,
    # the number of points that are less than or equal to the given threshold,
    # and the number of points that are more than the given threshold
    for index in range(0, len(threshold_coefficients)):
        if threshold_coefficients[index] <= threshold:
            wt_under += threshold_coefficients[index]
            no_under += 1
        elif threshold_coefficients[index] > threshold:
            wt_over += threshold_coefficients[index]
            no_over += 1

    # calculating the mean of the coefficients less than or equal to the splitting threshold
    # and conditions for division by zero
    if no_under == 0:
        mean_under = 0
    else:
        mean_under = wt_under / no_under

    # calculating the mean of the coefficients more than the splitting threshold
    # and conditions for division by zero
    if no_over == 0:
        mean_over = 0
    else:
        mean_over = wt_over / no_over

    # wt_under = the sum of threshold coefficients less than or equal to the given threshold / total sum of all coefficients
    wt_under = wt_under / total
    # wt_over = the sum of threshold coefficients more than to the given threshold / total sum of all coefficients
    wt_over = wt_over / total

    var_under = 0           # initializing the variance of the points less than or equal to the given threshold
    var_over = 0            # initializing the variance of the points more than the given threshold

    # calculating the sum of ((threshold coefficients less than or equal to the given threshold - their mean) ^ 2)
    # and the sum of ((threshold coefficients more than the given threshold - their mean) ^ 2)
    for index in range(0, len(threshold_coefficients)):
        if threshold_coefficients[index] <= threshold:
            var_under += math.pow((threshold_coefficients[index] - mean_under), 2)
        elif threshold_coefficients[index] > threshold:
            var_over += math.pow((threshold_coefficients[index] - mean_over), 2)

    # calculating var_under = sum of ((threshold coefficients less than or equal to the given threshold - their mean) ^ 2) / n - 1
    # and conditions for dividing by zero
    if (no_under - 1) == 0:
        var_under = var_under / no_under
    else:
        var_under = var_under / (no_under - 1)

    # calculating var_over = sum of ((threshold coefficients more than the given threshold - their mean) ^ 2) / n - 1
    # and conditions for dividing by zero
    if (no_over - 1) == 0:
        var_over = var_over / no_over
    else:
        var_over = var_over / (no_over - 1)

    # calculating the mixed variance and returning it
    mixed_variance = wt_under * var_under + wt_over * var_over

    return round(mixed_variance, 3)


def plot(threshold_coefficients, bin_size, best_threshold, best_new_threshold):
    '''
    function to plot the mixed variance as a function of the threshold
    :param threshold_coefficients: list of threshold coefficients of all the data points
    :param bin_size: the bin size for the data
    :param best_threshold: the threshold value for splitting the unripe and ripe data points
    :param nest_new_threshold: the threshold value for splitting the ripe and rotting data points
    '''

    variances = []          # list of mixed variances to be plotted along the y-axis
    thresholds = []         # list of threshold values to be plotted along the x-axis

    # using the "try them all" approach, calculating the mixed variances for thresholds at 0.05, 0.1, 0.15, 0.2 ...
    # and appending the threshold values and their mixed variances to their respective lists
    for threshold in np.arange(bin_size, threshold_coefficients[-1] + bin_size, bin_size):
        thresholds.append(threshold)
        mixed_variance = calculate_mixed_variance(threshold_coefficients, threshold)
        variances.append(mixed_variance)

    plt.plot(thresholds, variances)                                # thresholds v/s their corresponding variances
    plt.axis([0.05, thresholds[-1], 0, variances[-1]])             # axes scales

    # vertical red line at the threshold the separates the unripe melons on the left from the rest on the right of it
    plt.axvline(x=best_threshold, color="r", label="threshold for unripe/ripe")

    # vertical yellow line at the threshold the separates the rotting melons on the right from the rest on the left of it
    plt.axvline(x=best_new_threshold, color="y", label="threshold for ripe/rotting")

    plt.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0.)     # legend for the vertical lines
    plt.xlabel("thresholds")                                       # label for the x-axis
    plt.ylabel("variances")                                        # label for the y-axis
    plt.title("variance as a function of threshold")               # title of the graph
    plt.show()                                                     # displaying the graph

main()