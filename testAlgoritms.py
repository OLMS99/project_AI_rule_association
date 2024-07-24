from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import random
import math
import numpy as np
import time

import ActivationFunctions as ACT
import ANNTests
import LossFunctions as Loss
import ModelMetrics as metrics
import NeuralNetwork as NN
import Node
import Utils

import KT
import MofN
import RuleExtractionLearning as REL
import RxREN

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

def load_example(RNGseed):
    dataset = load_iris()
    data = dataset.data

    labelb = preprocessing.LabelBinarizer()
    label_target = labelb.fit_transform(dataset.target)
    classes = labelb.fit(dataset.target).classes_

    split_train_size = 0.7
    split_test_size = 1 - split_train_size

    train_X, valid_X, train_y, valid_y = train_test_split(data, label_target, test_size = split_test_size, random_state = 13)

    ANN = NN.nnf([4, 5, 3],[ACT.sigmoid, ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = RNGseed)
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
            neuron_info.append(params["b"+str(i+1)][j])
            neuron_info.append(params["f"+str(i+1)].__name__)
            neuron_layer.append(neuron_info)

        U.append(neuron_layer)

    return U

def algoritmo_1_KT(seed):
    ANN, C, DataX, DataY = load_example(seed)
    params = ANN.get_params()
    U = Neurons_to_Lists(params)
    result = KT.KT_1(U, debug=True)

    if len(result) > 0:
        for r in result:
            r.print()
    else:
        print("nenhuma regra feita")

def algoritmo_2_MofN(seed):
    ANN, _, DataX, Datay = load_example(seed)
    params = ANN.get_params()
    U = Neurons_to_Lists(params)
    result = MofN.MofN_2(U, ANN, DataX, Datay, debug=True)

    if len(result) > 0:
        for r in result:
            r.print()
    else:
        print("no rule made")

    #for case in DataX[0]:
        #tenta todas as arvores e ve os resultados
        #resposta = result.step(case)
        #compare

def algoritmo_3_RuleExtractLearning(seed):
    ANN, C, DataX, _ = load_example(seed)
    result = REL.Rule_extraction_learning_3(ANN, C, DataX[0], debug = True)
    for r in result.keys():
        if result[r]:
            print("rule made for %s" % (r))
            result[r].print()
        else:
            print("no rule made for %s" % (r))

    #for case in DataX[0]:
        #tenta todas as arvores e ve qual da true e qual da false
        #resposta = result.step(case)
        #compare

def algoritmo_4_RxRen(seed):
    ANN, C, DataX, Datay = load_example(seed)

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

    resultado = RxREN.RxREN_4(ANN, U, T, y, C, debug = True)

    if len(resultado) > 0:
        for r in resultado:
            r.print()
    else:
        print("nenhuma regra feita")
    #for case in T:
        #resposta = resultado.step(case)
        #compare

def load_wine_cobaia(random_state, split_train_size=0.7):
    dataset = load_wine()
    data = dataset.data

    labelb = preprocessing.LabelBinarizer()
    label_target = labelb.fit_transform(dataset.target)
    classes = labelb.fit(dataset.target).classes_

    split_test_size = 1 - split_train_size

    train_X,valid_X,train_y,valid_y = train_test_split(data, label_target, test_size=split_test_size, random_state=random_state)
    return classes, train_X, valid_X, train_y, valid_y

def load_wisconsin_cobaia(random_state, split_train_size=0.7):
    dataset = load_breast_cancer()
    data = dataset.data

    labelb = preprocessing.LabelBinarizer()
    label_target = labelb.fit_transform(dataset.target)
    classes = labelb.fit(dataset.target).classes_

    split_test_size = 1 - split_train_size

    train_X,valid_X,train_y,valid_y = train_test_split(data, label_target, test_size=split_test_size, random_state=random_state)
    return classes, train_X, valid_X, train_y, valid_y

def load_iris_cobaia(random_state, split_train_size=0.7):
    dataset = load_iris()
    data = dataset.data

    labelb = preprocessing.LabelBinarizer()
    label_target = labelb.fit_transform(dataset.target)
    classes = labelb.fit(dataset.target).classes_

    split_test_size = 1 - split_train_size

    train_X,valid_X,train_y,valid_y = train_test_split(data, label_target, test_size=split_test_size, random_state=random_state)
    return classes, train_X, valid_X, train_y, valid_y

def load_models_params(x_train, x_valid, y_train, y_valid, nEntrada, nSaida, seed, tamOculto, nLayers = 1, debug = False):
    if debug:
                print("x train: (%d, %d)" % (x_train.shape[0] ,x_train.shape[1]))
                print("y train: (%d, %d)" % (y_train.shape[0], y_train.shape[1]))
                print("x valid: (%d, %d)" % (x_valid.shape[0], x_valid.shape[1]))
                print("y valid: (%d, %d)" % (y_valid.shape[0], y_valid.shape[1]))

    criterios = tamOculto if isinstance(tamOculto, list) else [tamOculto]

    results = []
    if nLayers > 0:
        for caso in criterios:
            if debug:
                print("criterio atual: ", caso)

            ordem = []
            ordem.append(nEntrada)
            for i in range(nLayers):
                ordem.append(caso)
            ordem.append(nSaida)

            model = NN.nnf(ordem, [ACT.sigmoid]*(nLayers+1), Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
            model.train(x_train, y_train, x_valid, y_valid, epochs=1000, learning_rate=0.01)
            params = model.get_params()

            pred_train = np.zeros(shape=(x_train.shape[0], nSaida))
            pred_valid = np.zeros(shape=(x_valid.shape[0], nSaida))
            for i, sample in enumerate(x_train):
                pred_train[i] = model.predict(sample)

            for i, sample in enumerate(x_valid):
                pred_valid[i] = model.predict(sample)

            T, y = filter_correct_answers([x_train,x_valid],[y_train,y_valid],[pred_train,pred_valid])
            correct_cases = [T, y]
            acc = metrics.Compute_Acc_naive(pred_valid, y_valid)
            results.append([model, correct_cases, acc])

    else:
        model = NN.nnf([nEntrada, nSaida], [ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
        model.train(x_train, y_train, x_valid, y_valid, epochs=1000, learning_rate=0.01)
        params = model.get_params()

        pred_train = np.zeros(shape=(x_train.shape[0], nSaida))
        pred_valid = np.zeros(shape=(x_valid.shape[0], nSaida))
        for i, sample in enumerate(x_train):
            pred_train[i] = model.predict(sample)

        for i, sample in enumerate(x_valid):
            pred_valid[i] = model.predict(sample)

        correct_cases = filter_correct_answers([x_train,x_valid],[y_train,y_valid],[pred_train,pred_valid])
        acc = metrics.Compute_Acc_naive(pred_valid, y_valid)
        results.append([model, correct_cases, acc])

    return results

def test_algorithms(modelParamsList, dataBase, classes, debug = False):

    results = []
    for case in modelParamsList:
        model = case[0].copy()
        correct_cases = case[1]

        if debug: 
            print(Neurons_to_Lists(model.get_params()))

        algo1_result = KT.KT_1(Neurons_to_Lists(model.get_params()), debug = debug)
        algo2_result = MofN.MofN_2(Neurons_to_Lists(model.get_params()), model, dataBase[0], dataBase[1], debug = debug)
        algo3_result = REL.Rule_extraction_learning_3(model, classes, dataBase[0][1], debug = debug)

        if debug:
            print(correct_cases)

        algo4_result = RxREN.RxREN_4(model, Neurons_to_Lists(model.get_params()), correct_cases[0], correct_cases[1], classes, debug = debug)

        results.append([algo1_result, algo2_result, algo3_result, algo4_result])
    return results

def rule_predict(rules, inputVal):
    results = []

    for r in rules:
        results.append(r.step(inputVal))

    return results

def main_test():
    decisionTreeSeed = 42
    RNGseed = 1
    np.random.seed(RNGseed)

    #carregar exemplos e classes dos exemplos
    Iris_classes, X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid = load_iris_cobaia(RNGseed)
    Wine_classes, X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid = load_wine_cobaia(RNGseed)
    Wisconsin_classes, X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid = load_wisconsin_cobaia(RNGseed)

    Iris_Database = [[X_Iris_train, X_Iris_valid],[y_Iris_train, y_Iris_valid]]
    Wine_Database = [[X_Wine_train, X_Wine_valid],[y_Wine_train, y_Wine_valid]]
    Wisconsin_Database = [[X_Wisconsin_train, X_Wisconsin_valid],[y_Wisconsin_train, y_Wisconsin_valid]]

    #regras = [nEntrada, nEntrada+1, 2*nEntrada-1, 2*nEntrada, nSaida, nSaida+1, 2*nSaida-1, 2*nSaida, math.ceil((nEntrada+nSaida)/2), math.ceil((nEntrada*2+nSaida)/3)]
    tamOculto = []

    nEntradaIris = 4
    nSaidaIris = 3
    regrasIris = [nEntradaIris, nEntradaIris+1, 2*nEntradaIris-1, 2*nEntradaIris, nSaidaIris, nSaidaIris+1, 2*nSaidaIris-1, 2*nSaidaIris, math.ceil((nSaidaIris+nEntradaIris)/2), math.ceil((2*nSaidaIris+nEntradaIris)/3)]

    nEntradaWine = 13
    nSaidaWine = 3
    regrasWine = [nEntradaWine, nEntradaWine+1, 2*nEntradaWine-1, 2*nEntradaWine, nSaidaWine, nSaidaWine+1, 2*nSaidaWine-1, 2*nSaidaWine, math.ceil((nSaidaWine+nEntradaWine)/2), math.ceil((2*nSaidaWine+nEntradaWine)/3)]

    nEntradaWisconsin = 30
    nSaidaWisconsin = 2
    regrasWisconsin = [nEntradaWisconsin, nEntradaWisconsin+1, 2*nEntradaWisconsin-1, 2*nEntradaWisconsin, nSaidaWisconsin, nSaidaWisconsin+1, 2*nSaidaWisconsin-1, 2*nSaidaWisconsin, math.ceil((nSaidaWisconsin+nEntradaWisconsin)/2), math.ceil((2*nSaidaWisconsin+nEntradaWisconsin)/3)]

    #montar arvores de decisão
    decisionTree_Wine = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Wine.fit(X_Wine_train, y_Wine_train)

    decisionTree_Wisconsin = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Wisconsin.fit(X_Wisconsin_train, y_Wisconsin_train)

    decisionTree_Iris = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Iris.fit(X_Iris_train, y_Iris_train)

    #montar redes neurais

    #0 hidden layer

    #Wine_model_cases_n0 = load_models_params(X_Wine_train, X_Wine_valid, y_Wine_train,  y_Wine_valid, 13, 3, regrasWine, RNGseed, nLayers = 0, debug = True)
    #Wisconsin_model_cases_n0 = load_models_params(X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid, 30, 2, regrasWisconsin, RNGseed, nLayers = 0, debug = True)
    #Iris_model_cases_n0 = load_models_params(X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid, 4, 3, regrasIris, RNGseed, nLayers = 0, debug = True)

    #ruleSetsResults_0H_Wine = test_algorithms(Wine_model_cases_n0, Wine_Database, Wine_classes, debug = True)
    #ruleSetsResults_0H_Wisconsin = test_algorithms(Wisconsin_model_cases_n0, Wisconsin_Database, Wisconsin_classes, debug = True)
    #ruleSetsResults_0H_Iris = test_algorithms(Iris_model_cases_n0, Iris_Database, Iris_classes, debug = True)

    #1 hidden layer

    Wine_model_cases_n1 = load_models_params(X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid, 13, 3, regrasWine[3:6], RNGseed, debug = True)
    Wisconsin_model_cases_n1 = load_models_params(X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid, 30, 2, regrasWisconsin[3:6], RNGseed, debug = True)
    Iris_model_cases_n1 = load_models_params(X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid, 4, 3, regrasIris[3:6], RNGseed, debug = True)

    ruleSetsResults_1H_Wine = test_algorithms(Wine_model_cases_n1, Wine_Database, Wine_classes, debug = True)
    ruleSetsResults_1H_Wisconsin = test_algorithms(Wisconsin_model_cases_n1, Wisconsin_Database, Wisconsin_classes, debug = True)
    ruleSetsResults_1H_Iris = test_algorithms(Iris_model_cases_n1, Iris_Database, Iris_classes, debug = True)

    #2 hidden layers

    #Wine_model_cases_n2 = load_models_params(X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid, 13, 3, regrasWine, RNGseed, nLayers = 2, debug = True)
    #Wisconsin_model_cases_n2 = load_models_params(X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid, 30, 2, regrasWisconsin, RNGseed, nLayers = 2, debug = True)
    #Iris_model_cases_n2 = load_models_params(X_Iris_train, X_Iris_valid, y_Iris_train,  y_Iris_valid, 4, 3, regrasIris, RNGseed, nLayers = 2, debug = True)

    #ruleSetsResults_2H_Wine = test_algorithms(Wine_model_cases_n2, Wine_Database, Wine_classes, debug = True)
    #ruleSetsResults_2H_Wisconsin = test_algorithms(Wisconsin_model_cases_n2, Wisconsin_Database, Wisconsin_classes, debug = True)
    #ruleSetsResults_2H_Iris = test_algorithms(Iris_model_cases_n2, Iris_Database, Iris_classes, debug = True)

    #3 hidden layers

    #Wine_model_cases_n3 = load_models_params(X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid, 13, 3, regrasWine, RNGseed, nLayers = 3, debug = True)
    #Wisconsin_model_cases_n3 = load_models_params(X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid, 30, 2, regrasWisconsin, RNGseed, nLayers = 3, debug = True)
    #Iris_model_cases_n3 = load_models_params(X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid, 4, 3, regrasIris, RNGseed, nLayers = 3, debug = True)

    #ruleSetsResults_3H_Wine = test_algorithms(Wine_model_cases_n3, Wine_Database, Wine_classes, debug = True)
    #ruleSetsResults_3H_Wisconsin = test_algorithms(Wisconsin_model_cases_n3, Wisconsin_Database, Wisconsin_classes, debug = True)
    #ruleSetsResults_3H_Iris = test_algorithms(Iris_model_cases_n3, Iris_Database, Iris_classes, debug = True)

    #Avaliar Conjunto de regras resultantes



    return

def simpleTest():
    seed = 1
    algoritmo_1_KT(seed)
    algoritmo_2_MofN(seed)
    algoritmo_3_RuleExtractLearning(seed)
    algoritmo_4_RxREN(seed)
    print("bateria de teste terminado")
    return

#simpleTest()
main_test()