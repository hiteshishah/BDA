"""
hw07_hiteshi_shah.py
author: Hiteshi Shah
date: 10/29/2017
description: To find patterns using agglomerative clustering and draw a fendrogram
"""

import numpy as np
import math
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def main():
    # getting the training data from the CSV file
    data = np.genfromtxt("HW_AG_SHOPPING_CART_v500.csv", delimiter=",", dtype="unicode")

    # separating the attributes from the rest of the data
    attributes = data[0]
    data = data[1:].astype(np.int)

    # initializing the following dictionaries to keep track of the cluster ID of each data point
    # as well as the center of mass of each cluster
    centers_of_mass = {}                # cluster ID -> center of mass
    cluster_ids = {}                    # unique ID -> cluster ID
    # initially, each point's unique (guest) ID is its own cluster ID and its data is its center of mass
    for point in data:
        cluster_ids[point[0]] = point[0]
        centers_of_mass[point[0]] = point[1:]

    X = []                              # initializing the matrix to keep track of the last 30 clusters
    index = 1                           # initializing the index for printing the final 3 clusters

    # this loop runs until all the data points have been merged into one big cluster
    while len(set(cluster_ids.values())) != 1:
        minimum_distance = math.inf     # initializing the minimum distance to infinity
        # these nested loops compute the distance of each point from every other point
        for id1 in centers_of_mass.keys():
            for id2 in centers_of_mass.keys():
                # condition to avoid computing the distance from a cluster to itself
                if id1 == id2:
                    continue
                # computes the distance between id1 and id2
                distance = compute_euclidean_distance(centers_of_mass[id1], centers_of_mass[id2])
                # if the computed distance is smaller than the previous minimum distance,
                # update the minimum distance and assign the smaller cluster ID to unique_id_1
                # and larger cluster ID to unique_id_2
                if distance < minimum_distance:
                    minimum_distance = distance
                    unique_id_1 = min(id1, id2)
                    unique_id_2 = max(id1, id2)
        # once the cluster IDs with the shortest distance have been computed, merge them to the smaller cluster ID
        cluster_ids, centers_of_mass = merge_clusters(unique_id_1, unique_id_2, cluster_ids, centers_of_mass)

        # append the center of mass of the merged cluster to the matrix X
        X.append(centers_of_mass[unique_id_1])

        # if the current merged cluster is within the last 3 clusters, print out their center of mass
        if len(set(cluster_ids.values())) <= 3:
            print("Group " + str(index) + ": " + str(centers_of_mass[unique_id_1]))
            index += 1

    # performs hierarchical clustering on the matrix X which contains data of the last 30 clusters
    Z = linkage(X, method="centroid", metric="euclidean")
    # plots the hierarchical clustering as a dendrogram of the last 30 merges
    dendrogram(Z, truncate_mode="lastp", p=30)
    # displays the graph
    plt.show()

def compute_euclidean_distance(list1, list2):
    '''
    function to compute the euclidean distance between the data in list1 and list2
    :param list1: list of attribute values of the first data point
    :param list2: list of attribute values of the second data point
    :return: the euclidean distance between list1 and list2
    '''

    # the euclidean distance between list1 and list2 is the square root of the total sum of the squared difference
    # between each of the attribute values
    sum = 0
    for i in range(0, len(list1)):
        sum += math.pow(list1[i] - list2[i], 2)

    return math.pow(sum, 0.5)

def merge_clusters(unique_id_1, unique_id_2, cluster_ids, centers_of_mass):
    '''
    function to merge two clusters to the cluster with smaller ID
    :param unique_id_1: smaller cluster ID
    :param unique_id_2: larger cluster ID
    :param cluster_ids: dictionary that maps each data point's unique ID to its cluster ID
    :param centers_of_mass: dictionary that maps each data point's cluster ID to its center of mass
    :return: the updated dictionaries cluster_ids and centers_of_mass
    '''

    # fetching the cluster IDs and the centers of mass of the clusters to be merged
    cluster_id_1 = cluster_ids[unique_id_1]
    cluster_id_2 = cluster_ids[unique_id_2]
    center_of_mass_1 = centers_of_mass[cluster_id_1]
    center_of_mass_2 = centers_of_mass[cluster_id_2]

    # initializing the array for the new, computed center of mass
    new_center_of_mass = []

    # computing the new center of mass by taking the average of the two individual centers of mass
    for i in range(0, len(center_of_mass_1)):
        new_center_of_mass.append((center_of_mass_1[i] + center_of_mass_2[i]) / 2)

    # updating the center of the mass of the cluster with the smaller ID
    centers_of_mass[cluster_id_1] = new_center_of_mass
    # removing the cluster with the larger ID from the dictionary
    centers_of_mass.pop(cluster_id_2)
    # assigning the smaller cluster ID to the other cluster
    cluster_ids[unique_id_2] = cluster_ids[unique_id_1]
    # assigning the smaller cluster ID to every other data point that belongs to the other cluster
    for key, value in cluster_ids.items():
        if value == unique_id_2:
            cluster_ids[key] = cluster_ids[unique_id_1]

    return cluster_ids, centers_of_mass

main()