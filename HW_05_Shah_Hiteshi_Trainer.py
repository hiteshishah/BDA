"""
hw05_shah_hiteshi_trainer.py
author: Hiteshi Shah
date: 10/19/2017
description: To write a program that takes in the training data and in turn writes a classification program for the test data
"""

import numpy as np
import math

class TreeNode:
    """
    A tree node contains:
     :slot attr: The attribute that the node belongs to
     :slot val: The threshold value for splitting the node into left & right
     :slot left: The left child of the node
     :slot right: The right child of the node
     :slot child: This property indicates whether the child is a "Left" child or a "Right" child
    """
    __slots__ = 'attr', 'val', 'left', 'right', 'child'

    def __init__(self, attr, val, left=None, right=None, child=None):
        """
        function to initialize a node.
        :param attr: The attribute that the node belongs to
        :param val: The threshold value for splitting the node into left & right
        :param val: The threshold value for splitting the node into left & right
        :param left: The left child of the node
        :param right: The right child of the node
        :param child: This property indicates whether the child is a "Left" child or a "Right" child
        """
        self.attr = attr
        self.val = val
        self.left = left
        self.right = right
        self.child = child

def main():

    # getting the training data from the CSV file
    data = np.genfromtxt("HW_05C_DecTree_TRAINING__v540.csv", delimiter=",", dtype="unicode")

    # separating the attributes from the rest of the data
    attributes = data[0]
    attributes = attributes[:len(attributes) - 1]

    # separating the training data from the target variable and converting the training data to float
    training_data = data[1:].T
    target_variable = training_data[len(training_data) - 1]
    training_data = training_data[:len(training_data) - 1].astype(np.float)

    # running the decision tree on the training data
    root = decision_tree(training_data, attributes, target_variable)

    # setting the child labels in the decision tree
    setChild(root)

    # writing the classification program
    with open('HW_05_Shah_Hiteshi_Classifier.py', 'a') as file:
        # some boiler plate code at the start of the program that loads the test data into the classifier
        file.write('import numpy as np\n\n'
                   'data = np.genfromtxt("HW_05C_DecTree_TESTING__FOR_STUDENTS__v540.csv", delimiter=",", dtype="unicode")\n'
                   'attributes = data[0]\n'
                   'data = data[1:].astype(np.float)\n'
                   'classes = []\n'
                   'for line in data:\n')

        # recursive function that writes the if-else statements of the decision tree in the program
        preorder(root, file, 0, attributes)

        # code to save the results of the classifier program to a CSV file
        file.write('\nnp.savetxt("HW05_Shah_Hiteshi_MyClassifications.csv", classes, fmt="%s", delimiter=",", '
                   'header="Class", comments="")')

def decision_tree(training_data, attributes, target_variable):
    '''
    function that recursively builds the decision tree for the given training data
    :param training_data: the data on which the decision tree is built
    :param attributes: the attributes in the training data
    :param target_variable: list of values of the target variable for the classification, corresponding to the training data
    :return: returns the final decision tree
    '''

    # if the training data or the list of attributes is empty,
    # returns the majority of the target variable classes as the leaf node
    if (len(training_data) == 0) or (len(attributes) == 0):
        return TreeNode(find_majority(target_variable)[0], find_majority(target_variable)[0])

    # if all the target variable classes are the same,
    # returns the target variable class as the leaf node
    elif len(set(target_variable)) == 1:
        return TreeNode(target_variable[0], target_variable[0])

    # if there are less than or equal to 20 instances in target variable
    # returns the majority of the target variable classes as the leaf node
    elif len(target_variable) <= 20:
        return TreeNode(find_majority(target_variable)[0], find_majority(target_variable)[0])

    # recursively builds the decision tree
    else:
        # chooses the best attribute with the lowest gini index as the current root
        best_attribute, best_value = choose_attribute(attributes, training_data, target_variable)
        root = TreeNode(best_attribute, best_value)

        attribute_index = np.where(attributes == best_attribute)[0][0]

        # removes the chosen best attribute from the list of attributes
        attributes = np.delete(attributes, attribute_index)

        # transposing the training data to divide into left and right branches,
        # where values <= best value go to the left and values > best value go to the right
        training_data = training_data.T
        left_training_data = []
        right_training_data = []
        left_target_variable = []
        right_target_variable = []
        for index in range(0, len(training_data)):
            if training_data[index][attribute_index] <= best_value:
                left_training_data.append(training_data[index])
                left_target_variable.append(target_variable[index])
            else:
                right_training_data.append(training_data[index])
                right_target_variable.append(target_variable[index])
        left_training_data = np.array(left_training_data).T
        right_training_data = np.array(right_training_data).T

        # recursively calling the decision tree function on the left and right children
        left_child = decision_tree(left_training_data, attributes, left_target_variable)
        right_child = decision_tree(right_training_data, attributes, right_target_variable)

        # setting the left and right branches of the current root to the respective children and returning the resulting root
        root.left = left_child
        root.right = right_child
        return root

def find_majority(target_variable):
    '''
    function to find the majority of the classes in the given list of the target variable
    :param target_variable: list of classes of the target variable
    :return: the class with the maximum occurrence in the given list
    '''
    maxMap = {}         # dictionary for mapping the class with the maximum occurrence
    maximum = ('', 0)   # initializing the tuple for maximum(occurring element, no. of occurrences)

    # mapping all the classes in the list with their occurrences to the dictionary
    for value in target_variable:
        if value in maxMap:
            maxMap[value] += 1
        else:
            maxMap[value] = 1

        # keeping track of the maximum on the go
        if maxMap[value] > maximum[1]: maximum = (value, maxMap[value])

    return maximum

def choose_attribute(attributes, training_data, target_variable):
    '''
    function to choose the attribute with the lowest gini index
    :param attributes: list of attributes in the training data
    :param training_data: the training data
    :param target_variable: list of classes in the target variable, corresponding to the training data
    :return: the attribute (and its value) with the lowest gini index
    '''
    best_gini_index = math.inf          # initializing the best (lowest) gini index

    # for each attribute, splitting the training data into left & right at each value of the attribute
    # and computing the gini index at that value of the attribute
    for index in range(0, len(attributes)):
        for value in training_data[index]:
            left, right, sorted_target_variable = split(training_data[index], value, target_variable)
            gini_index = compute_gini_index(left, right, sorted_target_variable)

            # storing the lowest gini index, along with the attribute and its value
            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_attribute = attributes[index]
                best_value = value

    return best_attribute, best_value

def split(training_data, value, target_variable):
    '''
    function to split the given data, at the given value. Also to rearrange the classes in the list of the target variable
    :param training_data: the data to be split into left & right
    :param value: the value at which the split will occur
    :param target_variable: the list of classes of the target variable
    :return: left & right lists of the given data after the split, and the rearranged list of the target variable
    '''

    # rearranging the target variable in ascending order of the values in the given data
    sorted_target_variable = [x for _, x in sorted(zip(training_data, target_variable))]

    # sorting the given data in ascending order
    training_data = sorted(training_data)

    # initialzing the left and right lists for the split
    left = []
    right = []

    # for each value in the data, if value <= the given splitting value, we append that value to the left list
    # if value > the given splitting value, we append that value to the right list
    for data in training_data:
        if data <= value:
            left.append(data)
        else:
            right.append(data)

    return left, right, sorted_target_variable

def compute_gini_index(left, right, target_variable):
    '''
    function to compute the gini index, given the left and right splits and the target variable
    :param left: the left list after the split
    :param right: the right list after the split
    :param target_variable: the list of classes of the target variable
    :return: the computed gini index
    '''

    num_left = len(left)        # no. of values in the left list
    num_right = len(right)      # no. of values in the right list

    # initializing the counts for the classes of the target variable in both (left & right) lists
    left_greyhound_count = 0
    left_whippet_count = 0
    right_greyhound_count = 0
    right_whippet_count = 0

    # counting the occurrences of each class of the target variable in both (left & right) lists
    for index in range(0, num_left):
        if target_variable[index] == "Greyhound":
            left_greyhound_count += 1
        else:
            left_whippet_count += 1

    for index in range(0, num_right):
        if target_variable[index] == "Greyhound":
            right_greyhound_count += 1
        else:
            right_whippet_count += 1

    # computing the respective gini indexes of the left and right lists
    if left_greyhound_count + left_whippet_count == 0:
        left_gini_index = 0
    else:
        left_gini_index = 1 - math.pow(left_greyhound_count / (left_greyhound_count + left_whippet_count), 2)\
                      - math.pow(left_whippet_count / (left_greyhound_count + left_whippet_count), 2)

    if right_greyhound_count + right_whippet_count == 0:
        right_gini_index = 0
    else:
        right_gini_index = 1 - math.pow(right_greyhound_count / (right_greyhound_count + right_whippet_count), 2) \
                      - math.pow(right_whippet_count / (right_greyhound_count + right_whippet_count), 2)

    # computing the combined gini index
    gini_index = (num_left / (num_left + num_right)) * left_gini_index + (num_right / (num_left + num_right)) * right_gini_index

    return gini_index

def setChild(root, child=None):
    '''
    function to set "Left" and "Right" labels to child nodes
    :param root: the root of the decision tree
    :param child: the label indicating if the current node is a "Left" child or a "Right" child
    '''
    if root:
        root.child = child
        setChild(root.left, "Left")
        setChild(root.right, "Right")

def preorder(node, file, tabs, attributes):
    '''
    function to recursively write if-else statements of the decision tree, in pre-order style
    :param node: the current node in the decision tree
    :param file: the file to write the if-else statements into
    :param tabs: the current number of tabs (indentation)
    :param attributes: the list of attributes in the training data
    '''

    # returns if there is no node
    if not node:
        return

    # if the node is a leaf node, it writes code to append the class to the final list
    if node.attr == "Whippet" or node.attr == "Greyhound":
        if node.child == "Right":
            file.write("\t" * tabs)
            file.write("else:\n")
        file.write("\t" * (tabs + 1))
        file.write("classes.append('" + node.attr +"')\n")
    else:
        # if the node is a "Left" child, it writes the if statement
        if node.child != "Right":
            tabs += 1
            file.write("\t" * tabs)
            attr_index = np.where(attributes == node.attr)[0][0]
            file.write("if line[" + str(attr_index) + "] <= " + str(node.val) +":\n")

        # if the node is a "Right" child, it writes the else statement
        else:
            file.write("\t" * tabs)
            file.write("else:\n")
            if node.left:
                tabs += 1
                file.write("\t" * tabs)
                attr_index = np.where(attributes == node.attr)[0][0]
                file.write("if line[" + str(attr_index) + "] <= " + str(node.val) + ":\n")

    # recursively calling this function on the left and right branches of the current node
    preorder(node.left, file, tabs, attributes)
    preorder(node.right, file, tabs, attributes)

main()