from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import random
import math
import numpy as np
import time
import csv
import os

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

    train_X, valid_X, train_y, valid_y = train_test_split(data, label_target, test_size = split_test_size, random_state = RNGseed)

    ANN = NN.nnf([4, 5, 3],[ACT.sigmoid, ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = RNGseed)
    ANN.train(train_X, train_y, valid_X, valid_y, epochs=1000, learning_rate=0.01)

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
            for rule in r:
                rule.print()
    else:
        print("nenhuma regra feita")
    print(result)

def algoritmo_2_MofN(seed):
    ANN, _, DataX, Datay = load_example(seed)
    params = ANN.get_params()
    U = Neurons_to_Lists(params)
    result = MofN.MofN_2(U, ANN, DataX, Datay, debug=True)

    for r in result:
        if len(r) > 0:
            for rule in r:
                rule.print()
        else:
            print("no rule made")

    print(result)

def algoritmo_3_RuleExtractLearning(seed):
    ANN, C, DataX, _ = load_example(seed)
    result = REL.Rule_extraction_learning_3(ANN, C, DataX[0], debug = True)
    for label,ruleset in result.items():
        if ruleset is not None:
            print("rule made for %s" % (label))
            ruleset.print()
        else:
            print("no rule made for %s" % (label))

    print(result)

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

    for r in resultado.values():
        if len(r) <= 0:
            print("nenhuma regra feita")
        else:
            for rule in r:
                print(rule)

    print(resultado)

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
            results.append([model, correct_cases])

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
        results.append([model, correct_cases])

    return results

def test_algorithms(modelParamsList, dataBase, classes, debug = False):

    results = []
    for dataFrame in dataBase:
        result = []
        for case in modelParamsList:
            model = case[0].copy()
            correct_cases = case[1]

            if debug: 
                print(Neurons_to_Lists(model.get_params()))

            algo1_result = KT.KT_1(Neurons_to_Lists(model.get_params()))
            algo2_result = MofN.MofN_2(Neurons_to_Lists(model.get_params()), model, dataBase[0], dataBase[1])
            algo3_result = REL.Rule_extraction_learning_3(model, classes, dataBase[0][1])
            algo4_result = RxREN.RxREN_4(model, Neurons_to_Lists(model.get_params()), correct_cases[0], correct_cases[1], classes)

            result.append([algo1_result, algo2_result, algo3_result, algo4_result])
        results.append(result)
    return results

def parseRulesTest(model, ruleSets, X):
    pred_results = []

    for x_set in X:
        set_results = []

        for x_case in x_set:
            for x_row in x_case:

                KT_result = KT.parseRules(ruleSets[0], model, x_row)
                MofN_result = MofN.parseRules(ruleSets[1], model, x_row)
                REL_result = REL.parseRules(ruleSets[2], x_row)
                RxREN_result = RxRen.parseRules(ruleSets[3], x_row)

                results = [KT_result, MofN_result, REL_result, RxREN_result]
                set_results.append(results)

        pred_results.append(set_results)

    return pred_results

def compute_acc_rules_naive(ruleResults, y, classes):
    results = []
    for setIdx, ruleSetResults in enumerate(ruleResults):
        curr_ySet = y[setIdx]
        caseSize = curr_ySet.shape[0]
        totals = [0, 0, 0, 0]
        for idx, ruleCase in enumerate(ruleSetResults):
            print(ruleCase)
            pred = [np.round(result[0]) if len(result) > 0 else float('inf') for result in ruleCase]
            totals[0] += int(pred[0] == classes[curr_ySet[idx]])
            totals[1] += int(pred[1] == classes[curr_ySet[idx]])
            totals[2] += int(pred[2] == classes[curr_ySet[idx]])
            totals[3] += int(pred[3] == classes[curr_ySet[idx]])
        results.append(totals/caseSize)

    return results

def testesBateria(Database, Classes, numHLayers, entrada, saida, RNGseed, debug = False):
    #Database = [[X_train, X_valid],[y_train, y_valid]]
    regras = [entrada, entrada + 1, 2*entrada - 1, 2*entrada, saida, saida + 1, 2*saida - 1, 2*saida, math.ceil((saida + entrada)/2), math.ceil((2*entrada + saida)/3)]
    modelCases = load_models_params(Database[0][0], Database[0][1], Database[1][0], Database[1][1], entrada, saida, regras, RNGseed, debug = True)
    modelCasesAcc = [[model[0].accuracy(Database[0][0], Database[1][0]) for model in modelCases],[model[0].accuracy(Database[0][1], Database[1][1]) for model in modelCases]]

    print(modelCases)
    print(modelCasesAcc)

    ruleSetsResults = test_algorithms(modelCases, Database, Classes, debug = debug)
    rulePred = [parseRulesTest(model[0], ruleSet, Database) for model, ruleSet in zip(modelCases, ruleSetsResults)]
    ruleAcc = [compute_acc_rules_naive(pred, Database, Classes) for pred in rulePred]

    return [modelCasesAcc, ruleAcc]

def main_test():
    decisionTreeSeed = 42
    RNGseed = 1
    np.random.seed(RNGseed)

    os.mkdir("resultados")

    #carregar exemplos e classes dos exemplos
    Iris_classes, X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid = load_iris_cobaia(RNGseed)
    Wine_classes, X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid = load_wine_cobaia(RNGseed)
    Wisconsin_classes, X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid = load_wisconsin_cobaia(RNGseed)

    Iris_Database = [[X_Iris_train, X_Iris_valid],[y_Iris_train, y_Iris_valid]]
    Wine_Database = [[X_Wine_train, X_Wine_valid],[y_Wine_train, y_Wine_valid]]
    Wisconsin_Database = [[X_Wisconsin_train, X_Wisconsin_valid],[y_Wisconsin_train, y_Wisconsin_valid]]

    regras = ["E", "E + 1", "2E - 1", "2E", "S", "S + 1", "2S - 1", "2S", "(E + S)/2", "(2E + S)/3"]

    #montar arvores de decisão
    decisionTree_Wine = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Wine.fit(X_Wine_train, y_Wine_train)
    acc_decisionTree_Wine_train = metrics.Compute_Acc_naive(decisionTree_Wine.predict(X_Wine_train), y_Wine_train)
    acc_decisionTree_Wine_valid = metrics.Compute_Acc_naive(decisionTree_Wine.predict(X_Wine_valid), y_Wine_valid)

    decisionTree_Wisconsin = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Wisconsin.fit(X_Wisconsin_train, y_Wisconsin_train)
    acc_decisionTree_Wisconsin_train = metrics.Compute_Acc_naive(decisionTree_Wisconsin.predict(X_Wisconsin_train), y_Wisconsin_train)
    acc_decisionTree_Wisconsin_valid = metrics.Compute_Acc_naive(decisionTree_Wisconsin.predict(X_Wisconsin_valid), y_Wisconsin_valid)

    decisionTree_Iris = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Iris.fit(X_Iris_train, y_Iris_train)
    acc_decisionTree_Iris_train = metrics.Compute_Acc_naive(decisionTree_Iris.predict(X_Iris_train), y_Iris_train)
    acc_decisionTree_Iris_valid = metrics.Compute_Acc_naive(decisionTree_Iris.predict(X_Iris_valid), y_Iris_valid)

    with open('resultados/resultados_arvores.csv', 'w', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Database","Set","acuracy"])
        writer.writerow(["Wine", "train", acc_decisionTree_Wine_train])
        writer.writerow(["Wine", "valid", acc_decisionTree_Wine_valid])
        writer.writerow(["Wisconsin", "train", acc_decisionTree_Wisconsin_train])
        writer.writerow(["Wisconsin", "valid", acc_decisionTree_Wisconsin_valid])
        writer.writerow(["Iris", "train", acc_decisionTree_Iris_train])
        writer.writerow(["Iris", "valid", acc_decisionTree_Iris_valid])

    #1 hidden layer

    #Wine_1H_AccModelRule = testesBateria(Wine_Database, Wine_classes, 1, 13, 3, RNGseed, debug = True)
    #Wisconsin_1H_AccModelRule = testesBateria(Wisconsin_Database, Wisconsin_classes, 1, 30, 2, RNGseed, debug = True)
    Iris_1H_AccModelRule = testesBateria(Iris_Database, Iris_classes, 1, 4, 3, RNGseed, debug = True)

    with open('resultados/resultados_tests_1H.csv', 'w', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(regras)
    #    writer.writerows(Wine_1H_AccModelRule[0])
    #    writer.writerows(Wine_1H_AccModelRule[1])
    #    writer.writerows(Wisconsin_1H_AccModelRule[0])
    #    writer.writerows(Wisconsin_1H_AccModelRule[1])
    #    writer.writerows(Iris_1H_AccModelRule[0])
    #    writer.writerows(Iris_1H_AccModelRule[1])

    #2 hidden layers

    #Wine_2H_AccModelRule = testesBateria(Wine_Database, Wine_classes, 2, 13, 3, RNGseed, debug = True)
    #Wisconsin_2H_AccModelRule = testesBateria(Wisconsin_Database, Wisconsin_classes, 2, 30, 2, RNGseed, debug = True)
    Iris_2H_AccModelRule = testesBateria(Iris_Database, Iris_classes, 2, 4, 3, RNGseed, debug = True)

    with open('resultados/resultados_tests_2H.csv', 'w', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(regras)
    #    writer.writerows(Wine_2H_AccModelRule[0])
    #    writer.writerows(Wine_2H_AccModelRule[1])
    #    writer.writerows(Wisconsin_2H_AccModelRule[0])
    #    writer.writerows(Wisconsin_2H_AccModelRule[1])
    #    writer.writerows(Iris_2H_AccModelRule[0])
    #    writer.writerows(Iris_2H_AccModelRule[1])

    #3 hidden layers

    #Wine_3H_AccModelRule = testesBateria(Wine_Database, Wine_classes, 3, 13, 3, RNGseed, debug = True)
    #Wisconsin_3H_AccModelRule = testesBateria(Wisconsin_Database, Wisconsin_classes, 3, 30, 2, RNGseed, debug = True)
    Iris_3H_AccModelRule = testesBateria(Iris_Database, Iris_classes, 3, 4, 3, RNGseed, debug = True)

    with open('resultados/resultados_tests_3H.csv', 'w', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(regras)
    #    writer.writerows(Wine_3H_AccModelRule[0])
    #    writer.writerows(Wine_3H_AccModelRule[1])
    #    writer.writerows(Wisconsin_3H_AccModelRule[0])
    #    writer.writerows(Wisconsin_3H_AccModelRule[1])
    #    writer.writerows(Iris_3H_AccModelRule[0])
    #    writer.writerows(Iris_3H_AccModelRule[1])

    print("bateria de teste principal terminado")
    return

def simpleTest():
    seed = 1
    algoritmo_1_KT(seed)
    algoritmo_2_MofN(seed)
    algoritmo_3_RuleExtractLearning(seed)
    algoritmo_4_RxRen(seed)
    print("bateria de teste simples terminado")
    return

simpleTest()
main_test()