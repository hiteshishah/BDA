"""
hw00_shah_hiteshi.py
author: Hiteshi Shah
description: To find the maximum flow of cars per hour, as a function of speed in miles per hour
"""

import matplotlib.pyplot as plt
import math

def main():
    best_flux = -math.inf       # initializing the value of the best flux to - infinity
    best_speed = math.nan       # initializing the value of the best speed to NaN

    one_mile_in_feet = 5280
    reaction_time_human = 4     # in seconds
    car_length = 10             # in feet

    fluxs = []                  # list of values of flux corresponding to speed
    mphs = []                   # list of value of speed (ranging from 0 to 120 mph)

    # calculating the maximum flux from speed 0mph to 120mph
    # as well as storing the speed corresponding to the maximum flux
    for mph in range(0, 121):

        # computing car reaction time
        reaction_time_car = 0.00623 * math.pow(mph,2)

        # maximum value between human and car reaction times
        reaction_time = max(reaction_time_human, reaction_time_car)

        # compute the amount of road needed to react
        space_between_cars = reaction_time * mph * one_mile_in_feet / 60 / 60

        # compute distance from start of one car to the start of next
        space_allocation = car_length + space_between_cars

        # compute the cars per mile in the lane
        density = one_mile_in_feet / space_allocation

        # compute the cars per hour
        flux = density * mph

        # appending the current speed and the computed flux to their respective lists
        mphs.append(mph)
        fluxs.append(flux)

        if flux > best_flux:
            best_flux = flux
            best_speed = mph

    # printing the maximum flux and its corresponding speed
    print("Best Flux: " + str(best_flux) + " cars/hr")
    print("Best Speed: " + str(best_speed) + " miles/hr")

    # plot the graph of speed v/s flux
    plot(fluxs, mphs, best_flux, best_speed)


def plot(flux, mph, best_flux, best_speed):
    '''
    function to plot the of speed (mph) v/s flux (cars per hour)
    :param flux: along the y-axis, in cars per hour
    :param mph: along the x-axis, in miles per hour
    :param best_flux: maximum flux as a function of speed
    '''

    plt.plot(mph, flux, ":")                                    # speed v/s flux using a dotted line
    plt.axis([0, 120, 0, best_flux + 100])                      # axes scales
    plt.plot([best_speed, best_speed], [0, best_flux], "r")     # vertical red line from 0 to best_flux
    plt.plot([best_speed, 0], [best_flux, best_flux], "r")      # horizontal red line from 0 to best_speed
    plt.xlabel("Speed (miles/hour)")                            # label for the x-axis
    plt.ylabel("Flux (cars/hour)")                              # label for the y-axis
    plt.title("Road Efficiency as a function of speed")         # title of the graph
    plt.show()                                                  # displaying the graph

if __name__=="__main__":
    main()