from sklearn.datasets import load_iris
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import random
import math
import numpy as np
import time

import ActivationFunctions as ACT
import algorithms
import ANNTests
import LossFunctions as Loss
import ModelMetrics as metrics
import NeuralNetwork as NN
import Node
import Utils

import KT
import MofN

seed = 1
np.random.seed(seed)

def load_example():
    dataset = load_iris()
    data = dataset.data

    labelb = preprocessing.LabelBinarizer()
    label_target = labelb.fit_transform(dataset.target)
    classes = labelb.fit(dataset.target).classes_

    split_train_size = 0.7
    split_test_size = 1 - split_train_size

    train_X,valid_X,train_y,valid_y = train_test_split(data, label_target, test_size=split_test_size, random_state=13)

    ANN = NN.nnf([4, 5, 3],[ACT.sigmoid, ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
    ANN.train(train_X,train_y,valid_X,valid_y, epochs=1000, learning_rate=0.01)

    params = ANN.get_params()
    C = classes

    return ANN, C, [train_X,valid_X], [train_y, valid_y]

def Neurons_to_Lists(params):
    U=[]
    for i in range(params["num layers"]-1):
        neuron_layer = []
        for j in range(params["layer sizes"][i+1]):
            neuron_info = []
            neuron_info.append(params["W"+str(i+1)][j,:])
            neuron_info.append(np.squeeze(params["b"+str(i+1)])[j])
            neuron_info.append(params["f"+str(i+1)].__name__)
            neuron_layer.append(neuron_info)

        U.append(neuron_layer)

    return U

def algoritmo_1_KT():
    ANN, _, _, _ = load_example()
    params = ANN.get_params()
    U = Neurons_to_Lists(params)
    result = KT.KT_1(U, debug=True)

    if len(result) > 0:
        for r in result:
            r.print()
    else:
        print("no rule made")


def algoritmo_2_MofN():
    ANN, _, DataX, Datay = load_example()
    params = ANN.get_params()
    U = Neurons_to_Lists(params)
    result = MofN.MofN_2(U, ANN, DataX, Datay, debug=True)

    if len(result) > 0:
        for r in result:
            r.print()
    else:
        print("no rule made")


def algoritmo_3_RuleExtractLearning():
    ANN, C, DataX, _ = load_example()
    result = algorithms.Rule_extraction_learning_3(ANN, C, DataX[0], debug = True)
    for r in result.keys():
        if result[r]:
            result[r].print()
        else:
            print("no rule made for %s" % (r))

def generate_random_ruleTree(height=3, counter=0):
    treeNode = Node.Node(featureIndex=random.randint(0,10), threshold=random.uniform(0.,10.), negation=bool(random.getrandbits(1)))
    if counter <= height:
        treeNode.set_right(generate_random_ruleTree(height=height,counter=counter+1))
        treeNode.set_left(generate_random_ruleTree(height=height,counter=counter+1))

        #print("height: %d/%d" % (counter, height))
        #treeNode.right.print()
        #treeNode.left.print()

    return treeNode


def single_function_test():
    Ruletree = generate_random_ruleTree()
    #Ruletree.print()
    antecendents = Ruletree.getAntecedent()
    print(len(antecendents))
    random_deletion = random.choice(antecendents)
    #for premissa in antecendents:
    #    print("antecedente: %s" % (premissa[2]))
    #print("\npremissa a ser deletada: %s\n" % (random_deletion[2]))
    result = algorithms.filter(antecendents, random_deletion)
    print(len(result.getAntecedent()))

#algoritmo_1_KT()
#algoritmo_2_MofN() problema no tratamento de clusters
#algoritmo_3_RuleExtractLearning()
single_function_test()

