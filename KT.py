from itertools import combinations

import numpy as np
import Node
import Utils
import gc

#atribute->neuronios e seus resultados de saida (An)? ou antecedentes
#como representar-> coordenadas? (camada, ordem)?

#returns unused subsets of weightrs
def selectUnusedSubsets(attributes, size, threshold):
    final_selection = []

    for group in combinations(attributes, size):
        if sum_weights(group) > threshold:
            final_selection.append(group)

    return final_selection

def selectPositives(weights):
    PositiveLinks = list()

    for a, w in enumerate(weights):
        if w >= 0:
            positiveLink = (a, w)
            PositiveLinks.append(positiveLink)

    return PositiveLinks

def selectNegatives(weights):
    NegativeLinks = list()

    for a, w in enumerate(weights):
        if w < 0:
            negativeLink = (a, w)
            NegativeLinks.append(negativeLink)

    return NegativeLinks

def sum_weights(pesos):
    total = 0
    for item in pesos:
        total += item[1]

    return total

def makeRule_KT(layer_index, valorAprovado, valorRecusado, classe):
    premise1 = Node.Node(layerIndex = layer_index, featureIndex = valorAprovado[0], threshold = valorAprovado[1])
    premise2 = Node.Node(layerIndex = layer_index, featureIndex = valorRecusado[0], threshold = valorRecusado[1], negation = True)

    leaf = Node.Node(value = classe)

    premise1.set_right(premise2)
    premise2.set_right(leaf)
    return premise1

#TODO: implement algoritm to ease the complexity at line 111 to 121
def select_subsets(bigSet, threshold):
    pass

def KT_1(U, theta = 0, debug = False):
    R=[]

    for layer_idx, u in enumerate(U):
        layerRules = []
        for order_idx, neuron in enumerate(u):
            if debug:
                print("neuron: %s, layer: %s" % (order_idx, layer_idx + 1))
                print("info:", neuron)
                print("\n")

            neuron_weights = neuron[0] #array with the weights of a unit
            neuron_bias = neuron[1] #bias of the unit
            neuron_activation = neuron[2] #unit's activation function
            Su = selectPositives(neuron_weights)
            SumSu = sum_weights(Su)

            if debug:
                print("weights: %s\n bias: %s\nactivation: %s\n" % (neuron_weights, neuron_bias, neuron_activation))
                print("Sum of positives weights: %s" % (SumSu))
                print("theta: ", theta)

            if SumSu > theta:
                Sp = []
                for i in range(1, len(Su) + 1):
                    s = selectUnusedSubsets(Su, i, neuron_bias)
                    if debug:
                        print("num of possible subsets with size %s: %s" % (i, len(s)))
                    for subset in s:
                        if sum_weights(subset) > theta:
                            Sp.append(subset) #or
            else:
                continue

            if debug:
                print("Tamanho de Sp %s" % (len(Sp)))

            N = selectNegatives(neuron_weights)
            for p in Sp:
                N = selectNegatives(neuron_weights)
                if debug:
                    print("Tamanho de N %s" % (len(N)))
                    print("p: ", p)

                #Sum(p) + Sum(N-n) > bias of u >> Sum(p) + Sum(N) - Sum(n) > bias of u >> - Sum(n) > bias of u - (Sum(p) + Sum(N))
                #Sum(n) < Sum(p) + Sum(N) - bias of u
                WeightSumP = sum_weights(p)
                WeightSumN = sum_weights(N)
                comparison = WeightSumP + WeightSumN - neuron_bias
                for j in range(1, len(N) + 1):
                    filteredn = [comb for comb in combinations(N, j) if sum_weights(comb) < comparison]
                    for negSubset in filteredn:
                        WeightSumNegSub = sum_weights(negSubset)
                        if debug:
                            print("Sum p: %s\nSum N: %s\nSum of negSubset:%s" % (WeightSumP, WeightSumN, WeightSumNegSub))
                        for element in negSubset:
                            for item in p:
                                layerRules.append(makeRule_KT(layer_idx, item, element, (layer_idx + 1, order_idx)))
                    del filteredn
                    gc.collect()
        R.append(layerRules)

    return R

def combine_rules(R, numLayers):
    newRules = []

    for i in reversed(range(1, numLayers)):
        previousLayerRules = dict()

        for neuronRulePrv in R[i-1]:
            previousLayerLeaf = neuronRulePrv.right.right
            previousLayerRules[previousLayerLeaf.value[1]] = neuronRulePrv.right

        for neuronRuleCurr in R[i]:
            neuronFeature = neuronRuleCurr.getInputNeuron()[1]
            regraAnterior = previousLayerRules[neuronFeature]
            regraAnterior.set_right(neuronRuleCurr.copy())

    for r in R[0]:
        newRules.append(r)


    return newRules

def parseRules(ruleSet, model, inputValues):
    if len(ruleSet) is 0:
        return ()
    model.predict(inputValues)
    model_values = [np.squeeze(layer_val, axis=None) for layer_val in model.getAtributes()]
    results = ()
    noOutput = set(["no_output_values"])

    for idx, layerRules in enumerate(ruleSet):
        currResults = []
        if idx == 0:
            for rule in ruleSet[0]:
                if rule is None:
                    continue
                currResults.append(rule.step(model_values))
        else:
            for rule in layerRules:
                if rule is None:
                    continue
                if rule.getInputNeuron in results:
                    currResults.append(rule.step(model_values))

        results = set(currResults)
        results = results - noOutput
        results = list(results)

    return results if len(results) > 0 else ("no_results")

def isComplete(KTruleSet):
    for layerRules in KTruleSet:
        if len(layerRules) <= 0:
            return False
    return True

def delete(KTRuleSet):
    for layerRules in KTRuleSet:
        for rule in layerRules:
            if rule is None:
                continue
            rule.destroy()

def printRules(KTRuleSet):
    if len(KTRuleSet) > 0:
        for r in result:
            for rule in r:
                rule.print()
    else:
        print("nenhuma regra feita")
    print(KTRuleSet)