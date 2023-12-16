import itertools

import random
import math
import numpy as np
import pandas as pd 
import time

import ActivationFunctions
import LossFunctions
import ModelMetrics
import NeuralNetwork
import Node
import Utils
#atribute->neuronios e seus resultados de saida (An)? ou antecedentes
#como representar-> coordenadas? (camada, ordem)?

#returns unused subsets of weightrs
def selectUnusedSubsets(attributes, size, threshold):
    possible_sets = list(itertools.combinations(attributes, size))
    final_selection = []

    for group in possible_sets:
        if sum(group) >= threshold:
            final_selection.append(group)

    return final_selection

def selectPositives(weights):
    PositiveLinks = list()

    for a, w in enumerate(weights):
        if w >= 0:
            positiveLink = [a, w]
            PositiveLinks.append(a)

    return PositiveLinks

def selectNegatives(weights):
    NegativeLinks = list()

    for a, w in enumerate(weights):
        if w < 0:
            negativeLink = [a, w]
            NegativeLinks.append(a)

    return NegativeLinks

def KT_1(U, theta = 0, debug = False):
    R=[]

    for layer_idx, u in enumerate(U):
        for order_idx, neuron in enumerate(u):
            if debug:
                print("neuron: %s, layer: %s\n" % (order_idx, layer_idx))

            neuron_weights = neuron[0] #array with the weights of a unit
            neuron_bias = neuron[1] #bias of the unit
            neuron_activation = neuron[2] #unit's activation function
            Su = selectPositives(neuron_weights)
            SumSu = sum(Su)

            if debug:
                print("weights: %s\n bias: %s\nactivation:%s\n" % (neuron_weights, neuron_bias, neuron_activation.__name__))
                print("Sum of positives weights: %s" % (SumSu))

            if SumSu > theta:
                Sp = []
                for i in range(1, len(Su) + 1):
                    s = selectUnusedSubsets(Su, i, theta)
                    for subset in s:
                        if sum(subset) > theta:
                            Sp.append(subset) #or
            else:
                continue

            N = selectNegatives(neuron_weights)
            for p in Sp:
                N = selectNegatives(neuron_weights)
                for j in range(1, len(N) + 1):
                    n= list(itertools.combinations(N, j))
                    for negSubset in n:
                        if debug:
                            print("Sum p: %s\nSum N: %s\nSum of negSubset:%s" % (sum(p), sum(N), sum(negSubset)))
                        if sum(p) + (sum(N) - sum(negSubset)) > theta:
                            for element in negSubset:
                                for item in p:
                                    premise1 = Node.Node(layerIndex = layer_idx+1, featureIndex = order_idx, threshold = item)
                                    premise2 = Node.Node(layerIndex = layer_idx+1, featureIndex = order_idx, threshold = element, negation = True)

                                    leaf = Node.Node(value = neuron)

                                    premise1.set_right(premise2)
                                    premise2.set_right(leaf)
                                    R.append(premise1)

    return R