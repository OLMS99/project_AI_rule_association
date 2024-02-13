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
import RuleExtractionLearning as REL

seed = 1
np.random.seed(seed)

def filter_correct_answers(dataset, y, prediction):
    tamLinha_X = dataset[0].shape[1]
    tamLinha_y = y[0].shape[1]
    tamLinha_pred = len(prediction[0][0])

    #print("dataset 0 shape: (%d, %d)" % (dataset[0].shape[0], dataset[0].shape[1]))
    #print("dataset 1 shape: (%d, %d)" % (dataset[1].shape[0], dataset[1].shape[1]))
    #print("y 0 shape: (%d, %d)" % (y[0].shape[0], y[0].shape[1]))
    #print("y 1 shape: (%d, %d)" % (y[1].shape[0], y[1].shape[1]))
    #print("prediction 0 shape: %d" % (len(prediction[0])))
    #print("prediction 1 shape: %d" % (len(prediction[1])))
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #print(prediction[0])
    #print(prediction[1])

    dataX = np.append(dataset[0], dataset[1]).reshape(-1, tamLinha_X)
    datay = np.append(y[0], y[1]).reshape(-1, tamLinha_y)
    predictions_cases = np.append(prediction[0], prediction[1]).reshape(-1, tamLinha_pred)

    #print("size of dataX: %d" % (len(dataX)))
    #print("size of datay: %d" % (len(datay)))
    #print("size of predictions_cases: %d" % (len(predictions_cases)))
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #print(predictions_cases)

    comparison = []
    for i in range(len(dataX)):
        comparison.append(np.argmax(datay[i]) == np.argmax(predictions_cases[i]))

    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #print(comparison)

    returnDataX = dataX[comparison]
    returnDatay = datay[comparison]

    return returnDataX, returnDatay

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
    result = REL.Rule_extraction_learning_3(ANN, C, DataX[0], debug = True)
    for r in result.keys():
        if result[r]:
            print("rule made for %s" % (r))
            result[r].print()
        else:
            print("no rule made for %s" % (r))

def algoritmo_4_RxRen():
    ANN, C, DataX, Datay = load_example()

    params = ANN.get_params()
    U = Neurons_to_Lists(params)

    predictions = [[],[]]
    for case in DataX[0]:
        predictions[0].append(ANN.predict(case))
    row_size = len(predictions[0][0])
    predictions[0] = np.concatenate(predictions[0], axis=0).reshape(-1, row_size)
    for case in DataX[1]:
        predictions[1].append(ANN.predict(case))
    row_size = len(predictions[1][0])
    predictions[1] = np.concatenate(predictions[1], axis=0).reshape(-1, row_size)

    T, y = filter_correct_answers(DataX, Datay, predictions)

    resultado = algorithms.RxREN_4(ANN, U, T, y, C)

def generate_random_ruleTree(height=2, counter=0):
    treeNode = Node.Node(featureIndex=random.randint(0,10), threshold=random.uniform(0.,10.), negation=bool(random.getrandbits(1)))
    if counter <= height:
        treeNode.set_right(generate_random_ruleTree(height=height,counter=counter+1))
        treeNode.set_left(generate_random_ruleTree(height=height,counter=counter+1))

        #print("height: %d/%d" % (counter, height))
        #treeNode.right.print()
        #treeNode.left.print()

    return treeNode

def generate_static_ruleTree():
    height_0 = Node.Node(featureIndex=0, layerIndex=3, threshold=1.5, comparison="!=", negation=False)

    height_1 = [
        Node.Node(featureIndex=1, layerIndex=2, threshold=3.7, comparison=">=", negation=False),
        Node.Node(featureIndex=2, layerIndex=1, threshold=4.2, comparison="<=", negation=True)
    ]

    height_2 = [
        Node.Node(featureIndex=3, layerIndex=3, threshold=2.8, comparison=">", negation=True),
        Node.Node(featureIndex=4, layerIndex=2, threshold=5.9, comparison="<", negation=False),
        Node.Node(featureIndex=5, layerIndex=1, threshold=6.3, comparison="=", negation=True),
        Node.Node(featureIndex=6, layerIndex=3, threshold=9.6, comparison="!=", negation=True)
    ]

    height_1[0].set_left(height_2[0])
    height_1[0].set_right(height_2[1])
    height_1[1].set_left(height_2[2])
    height_1[1].set_right(height_2[3])

    height_0.set_left(height_1[0])
    height_0.set_right(height_1[1])

    return height_0

def single_function_test():
    Ruletree = generate_static_ruleTree()
    copia = Ruletree.copy_tree()

    antecendents = Ruletree.getAntecedent()

    random_deletion_a = random.choice(antecendents)
    lado_escolhido = random_deletion_a[0]

    if lado_escolhido == 1:
        random_deletion_b = random_deletion_a[1].right
    if lado_escolhido == -1:
        random_deletion_b = random_deletion_a[1].left
    else:
        random_deletion_b = antecendents[-2][1]


    copied_antecendents = copia.getAntecedent()
    print(len(copied_antecendents))

    result = REL.filter(antecendents, random_deletion_a[2])
    print(len(result.getAntecedent()))
    result = REL.filter(antecendents, random_deletion_a[2])
    print(len(result.getAntecedent()))

#algoritmo_1_KT()
#algoritmo_2_MofN() #problema no tratamento de clusters
#algoritmo_3_RuleExtractLearning()
algoritmo_4_RxRen()
#single_function_test()

