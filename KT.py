from itertools import combinations

import Node
import Utils

#atribute->neuronios e seus resultados de saida (An)? ou antecedentes
#como representar-> coordenadas? (camada, ordem)?

#returns unused subsets of weightrs
def selectUnusedSubsets(attributes, size, threshold):
    possible_sets = list(combinations(attributes, size))
    final_selection = []

    for group in possible_sets:
        if sum_weights(group) < threshold:
            final_selection.append(group)

    return final_selection

def selectPositives(weights):
    PositiveLinks = list()

    for a, w in enumerate(weights):
        if w >= 0:
            positiveLink = [a, w]
            PositiveLinks.append(positiveLink)

    return PositiveLinks

def selectNegatives(weights):
    NegativeLinks = list()

    for a, w in enumerate(weights):
        if w < 0:
            negativeLink = [a, w]
            NegativeLinks.append(negativeLink)

    return NegativeLinks

def sum_weights(pesos):
    total = 0
    for item in pesos:
        total += item[1]

    return total

def makeRule_KT(layer_index, order_index, valorAprovado, valorRecusado, classe):
    premise1 = Node.Node(layerIndex = layer_index, featureIndex = valorAprovado[0], threshold = valorAprovado[1])
    premise2 = Node.Node(layerIndex = layer_index, featureIndex = valorRecusado[0], threshold = valorRecusado[1], negation = True)

    leaf = Node.Node(value = classe)

    premise1.set_right(premise2)
    premise2.set_right(leaf)
    return premisse1

def KT_1(U, theta = 0, debug = False):
    R=[]

    for layer_idx, u in enumerate(U):
        for order_idx, neuron in enumerate(u):
            if debug:
                print("neuron: %s, layer: %s\n" % (order_idx, layer_idx + 1))

            neuron_weights = neuron[0] #array with the weights of a unit
            neuron_bias = neuron[1] #bias of the unit
            neuron_activation = neuron[2] #unit's activation function
            Su = selectPositives(neuron_weights)
            SumSu = sum_weights(Su)

            #if debug:
            #    print("weights: %s\n bias: %s\nactivation: %s\n" % (neuron_weights, neuron_bias, neuron_activation))
            #    print("Sum of positives weights: %s" % (SumSu))

            if SumSu > theta:
                Sp = []
                for i in range(1, len(Su) + 1):
                    s = selectUnusedSubsets(Su, i, neuron_bias)
                    for subset in s:
                        if sum_weights(subset) > theta:
                            Sp.append(subset) #or
            else:
                continue

            N = selectNegatives(neuron_weights)
            for p in Sp:
                N = selectNegatives(neuron_weights)
                for j in range(1, len(N) + 1):
                    n = list(combinations(N, j))
                    for negSubset in n:
                        if debug:
                            print("Sum p: %s\nSum N: %s\nSum of negSubset:%s" % (sum_weights(p), sum_weights(N), sum_weights(negSubset)))
                        if sum_weights(p) + (sum_weights(N) - sum_weights(negSubset)) > neuron_bias:
                            for element in negSubset:
                                for item in p:
                                    R.append(makeRule_KT(layer_idx, order_idx, item, element, (layer_idx + 1, order_idx)))
    return R