"""
hw_kmeans_shah_hiteshi.py
author: Hiteshi Shah
description: To find the number of clusters in the given data using k-means
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    # getting the data from the CSV file
    data = np.genfromtxt("HW_2171_KMEANS_DATA__v502.csv", delimiter=",", skip_header=True)

    n = len(data)           # the total number of data points in the given dataset
    num_of_clusters = []    # initializing the list of number of clusters
    cluster_sse = []        # initializing the list of SSEs (sum of squared errors) for each of the number of clusters
    k = 1                   # initializing the number of clusters k to 1

    # we run this loop for number of clusters k = 1 to 15
    while k <= 20:
        clusters = {}  # initializing the dictionary of clusters for the current k as follows:
        # cluster number -> centroid, data points in cluster, sse
        old_centroids = []  # initializing the list of the current centroids of the clusters
        num_of_clusters.append(k)  # appending the current value of k to the list of number of clusters

        # for every cluster in k, appending the initial seed points to the clusters dictionary as well as the
        # list of current centroids of the clusters
        for i in range(0, k):
            clusters[i] = [data[i + 10], [], None]
            old_centroids.append(data[i + 10])

        for _ in range(0, 1000):

            # computing the distance of every point to the centroids of each cluster and assigning the data point
            # to the cluster with the minimum distance to its centroid
            for j in range(0, n):
                minimum_distance = math.inf
                cluster = math.nan
                for l in range(0, len(clusters)):
                    distance = math.sqrt(math.pow(data[j][0] - clusters[l][0][0], 2) +
                                         math.pow(data[j][1] - clusters[l][0][1], 2) +
                                         math.pow(data[j][2] - clusters[l][0][2], 2))
                    if distance < minimum_distance:
                        minimum_distance = distance
                        cluster = l
                clusters[cluster][1].append(data[j])

            new_centroids = []         # initializing the list of the new centroids of the clusters

            # computing the new centroids of each of the clusters
            for ind in range(0, len(clusters)):
                new_centroid = [sum(x) for x in zip(*clusters[ind][1])]
                cluster_len = len(clusters[ind][1])
                new_centroid = [x / cluster_len for x in new_centroid]
                new_centroid = np.around(new_centroid, decimals=2)
                new_centroids.append(new_centroid)

            if np.array_equal(old_centroids, new_centroids):
                # stop if the newly computed centroids as the same as the old centroids of the clusters
                break
            else:
                # otherwise, assign the newly computed centroids as the current centroids of the clusters
                old_centroids[:] = new_centroids[:]
                for ind in range(0, len(clusters)):
                    clusters[ind][0] = new_centroids[ind]
                    clusters[ind][1] = []

        if k == 7:
            # plot the clusters when k = 7
            plot_clusters(clusters)

        sse = 0     # initializing the value of the sum of squared errors

        # compute the sum of squared errors of each of the clusters in k and summing them up
        # SSE = summation of (data point - centroid) ^ 2 for every data point in the cluster
        # repeating the above formula in every cluster and summing all the SSEs gives the final SSE
        for num in range(0, len(clusters)):
            centroid = clusters[num][0]
            cluster_data = clusters[num][1]
            sum_sse = 0
            for index in range(0, len(cluster_data)):
                sum_sse += math.pow(cluster_data[index][0] - centroid[0], 2) + \
                           math.pow(cluster_data[index][1] - centroid[1], 2) + \
                           math.pow(cluster_data[index][2] - centroid[2], 2)
            sse += sum_sse

        # appending the value of the SSE to the list of SSEs, corresponding to the current k
        clusters[cluster][2] = sse
        cluster_sse.append(sse)

        # incrementing the value of k, i.e., the number of clusters
        k += 1

    # plotting the graph of number of clusters v's their SSEs
    plot(num_of_clusters, cluster_sse)

def plot(num_of_clusters, cluster_sse):
    '''
    function to plot the graph of number of clusters against their SSEs
    :param num_of_clusters: list of number of clusters k
    :param cluster_sse: list of SSEs corresponding to each k
    '''
    plt.plot(num_of_clusters, cluster_sse, ":")     # number of clusters v/s their SSEs using a dotted line

    # changing the tick frequency along the x axis to display all values of k and their corresponding SSEs
    plt.xticks(np.arange(min(num_of_clusters), max(num_of_clusters) + 1, 1.0))

    plt.plot(7, cluster_sse[6], 'ro')               # marker indicating the best value of k
    print("SSE at k = 7: " + str(cluster_sse[6]))

    plt.xlabel("Number of clusters")                # label for the x-axis
    plt.ylabel("Sum of squared errors")             # label for the y-axis
    plt.show()                                      # displaying the graph

def plot_clusters(clusters):
    '''
    function to plot the graph of the given number of clusters
    :param clusters: dictionary of clusters
    '''
    fig = plt.figure()

    # projecting the graph in 3D since there are three attributes in the given dataset
    ax = fig.add_subplot(111, projection='3d')

    x = []                                              # initializing the list of x-coordinates of each of the clusters
    y = []                                              # initializing the list of y-coordinates of each of the clusters
    z = []                                              # initializing the list of z-coordinates of each of the clusters
    colors = ['b', 'y', 'r', 'g', 'c', 'm', 'k']        # initializing the list of colors for each of the clusters

    # plots the scatterplot for each cluster, in a different color
    for i in range(0, len(clusters)):
        data = clusters[i][1]
        data = np.array(data).T
        x.append(data[0])
        y.append(data[1])
        z.append(data[2])

        ax.scatter(x[i], y[i], z[i], c=colors[i], marker='o')

    ax.set_xlabel('Attrib01')                     # label for the x-axis
    ax.set_ylabel('Attrib02')                     # label for the y-axis
    ax.set_zlabel('Attrib03')                     # label for the z-axis

    plt.title("K-Means result with 7 clusters")   # title of the graph
    plt.show()                                    # displaying the graph

main()