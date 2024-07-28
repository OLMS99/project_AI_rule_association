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

def algoritmo_2_MofN(seed):
    ANN, _, DataX, Datay = load_example(seed)
    params = ANN.get_params()
    U = Neurons_to_Lists(params)
    result = MofN.MofN_2(U, ANN, DataX, Datay, debug=True)

    if len(result) > 0:
        for r in result:
            for rule in r:
                rule.print()
    else:
        print("no rule made")

    for case in DataX[0]:
        #tenta todas as arvores e ve os resultados
        #resposta = MofN.parseRules(result, )
        #compare
        pass

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

def getModelAcc(model, X, y):
    predictions = []

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
        algo4_result = RxREN.RxREN_4(model, Neurons_to_Lists(model.get_params()), correct_cases[0], correct_cases[1], classes, debug = debug)

        results.append([algo1_result, algo2_result, algo3_result, algo4_result])
    return results

def parseRulesTest(model, ruleSets, X):
    pred_results = []

    for x_case in X:
        model.predict(x_case)
        input_A = model.getAtributes()

        KT_result = KT.parseRules(ruleSets[0], input_A)
        MofN_result = MofN.parseRules(ruleSets[1], input_A)
        REL_result = REL.parseRules(ruleSets[2], x_case)
        RxREN_result = RxRen.parseRules(ruleSets[3], x_case)

        results = [KT_result, MofN_result, REL_result, RxREN_result]
        pred_results.append(results)

    return pred_results

def computer_acc_rules_naive(ruleResults, y, classes):
    caseSize = y.shape[0]
    totals = [0,0,0,0]
    for idx, ruleCase in enumerate(ruleResults):
        totals[0] += int(ruleCase[0] == classes[y[idx]])
        totals[1] += int(ruleCase[1] == classes[y[idx]])
        totals[2] += int(ruleCase[2] == classes[y[idx]])
        totals[3] += int(ruleCase[3] == classes[y[idx]])

    return totals / caseSize

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

    #montar redes neurais

    #0 hidden layer

    #Wine_model_cases_n0 = load_models_params(X_Wine_train, X_Wine_valid, y_Wine_train,  y_Wine_valid, 13, 3, regrasWine, RNGseed, nLayers = 0, debug = True)
    #Wisconsin_model_cases_n0 = load_models_params(X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid, 30, 2, regrasWisconsin, RNGseed, nLayers = 0, debug = True)
    #Iris_model_cases_n0 = load_models_params(X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid, 4, 3, regrasIris, RNGseed, nLayers = 0, debug = True)

    #ruleSetsResults_0H_Wine_train = test_algorithms(Wine_model_cases_n0, X_Wine_train, Wine_classes, debug = True)
    #ruleSetsResults_0H_Wisconsin_train = test_algorithms(Wisconsin_model_cases_n0, X_Wisconsin_train, Wisconsin_classes, debug = True)
    #ruleSetsResults_0H_Iris_train = test_algorithms(Iris_model_cases_n0, X_Iris_train, Iris_classes, debug = True)

    #ruleSetsResults_0H_Wine_valid = test_algorithms(Wine_model_cases_n0, X_Wine_valid, Wine_classes, debug = True)
    #ruleSetsResults_0H_Wisconsin_valid = test_algorithms(Wisconsin_model_cases_n0, X_Wisconsin_valid, Wisconsin_classes, debug = True)
    #ruleSetsResults_0H_Iris_valid = test_algorithms(Iris_model_cases_n0, X_Iris_valid, Iris_classes, debug = True)

    #ruleSetsResults_0H_Wine = [ruleSetsResults_0H_Wine_train, ruleSetsResults_0H_Wine_valid]
    #ruleSetsResults_0H_Wisconsin = [ruleSetsResults_0H_Wisconsin_train, ruleSetsResults_0H_Wisconsin_valid]
    #ruleSetsResults_0H_Iris = [ruleSetsResults_0H_Iris_train, ruleSetsResults_0H_Iris_valid]

    #for idx in range(len(ruleSetResults_0H_Wine)):
    #   get predictions of ruleSets and the accuracy
    #   parseRulesTest(model, ruleSetsResults_0H_Wine, Wine_Database[0])
    #   computer_acc_rules_naive(ruleSetsResults_0H_Wine, Wine_Database, Wine_classes)
    #   computer_acc_rules_naive(ruleSetsResults_0H_Wisconsin, Wisconsin_Database, Wisconsin_classes)
    #   computer_acc_rules_naive(ruleSetsResults_0H_Iris, Iris_Database, Iris_classes)
	
	#acc_models_Wine_n0 = [[],[]]
	#acc_models_Wisconsin_n0 = [[],[]]
	#acc_models_Iris_n0 = [[],[]]
	#
	#for idx in range(len(Wine_model_cases_n0)):
	#	acc_models_Wine_n0[0].append(metrics.compute_acc_naive(Wine_model_cases_n0[idx].predict(X_Wine_train),y_Wine_train))
	#	acc_models_Wine_n0[1].append(metrics.compute_acc_naive(Wine_model_cases_n0[idx].predict(X_Wine_valid),y_Wine_valid))
	#	acc_models_Wisconsin_n0[0].append(metrics.compute_acc_naive(Wisconsin_model_cases_n0[idx].predict(X_Wisconsin_train),y_Wisconsin_train))
	#	acc_models_Wisconsin_n0[1].append(metrics.compute_acc_naive(Wisconsin_model_cases_n0[idx].predict(X_Wisconsin_valid),y_Wisconsin_valid))
	#	acc_models_Iris_n0[0].append(metrics.compute_acc_naive(Iris_model_cases_n0[idx].predict(X_Iris_train),y_Iris_train))
	#	acc_models_Iris_n0[1].append(metrics.compute_acc_naive(Iris_model_cases_n0[idx].predict(X_Iris_valid),y_Iris_Valid))

    #1 hidden layer

    Wine_model_cases_n1 = load_models_params(X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid, 13, 3, regrasWine[3:6], RNGseed, debug = True)
    Wisconsin_model_cases_n1 = load_models_params(X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid, 30, 2, regrasWisconsin[3:6], RNGseed, debug = True)
    Iris_model_cases_n1 = load_models_params(X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid, 4, 3, regrasIris[3:6], RNGseed, debug = True)

    ruleSetsResults_1H_Wine_train = test_algorithms(Wine_model_cases_n1, X_Wine_train, Wine_classes, debug = True)
    ruleSetsResults_1H_Wisconsin_train = test_algorithms(Wisconsin_model_cases_n1, X_Wisconsin_train, Wisconsin_classes, debug = True)
    ruleSetsResults_1H_Iris_train = test_algorithms(Iris_model_cases_n1, X_Iris_train, Iris_classes, debug = True)

    ruleSetsResults_1H_Wine_valid = test_algorithms(Wine_model_cases_n1, X_Wine_valid, Wine_classes, debug = True)
    ruleSetsResults_1H_Wisconsin_valid = test_algorithms(Wisconsin_model_cases_n1, X_Wisconsin_valid, Wisconsin_classes, debug = True)
    ruleSetsResults_1H_Iris_valid = test_algorithms(Iris_model_cases_n1, X_Iris_valid, Iris_classes, debug = True)

    ruleSetsResults_1H_Wine = [ruleSetsResults_1H_Wine_train, ruleSetsResults_1H_Wine_valid]
    ruleSetsResults_1H_Wisconsin = [ruleSetsResults_1H_Wisconsin_train, ruleSetsResults_1H_Wisconsin_valid]
    ruleSetsResults_1H_Iris = [ruleSetsResults_1H_Iris_train, ruleSetsResults_1H_Iris_valid]

   #for idx in range(len(ruleSetResults_1H_Wine)):
    #   get predictions of ruleSets and the accuracy
    #   parseRulesTest(model, ruleSetsResults_1H_Wine, Wine_Database[0])
    #   computer_acc_rules_naive(ruleSetsResults_1H_Wine, Wine_Database, Wine_classes)
    #   computer_acc_rules_naive(ruleSetsResults_1H_Wisconsin, Wisconsin_Database, Wisconsin_classes)
    #   computer_acc_rules_naive(ruleSetsResults_1H_Iris, Iris_Database, Iris_classes)
	
	#acc_models_Wine_n1 = [[],[]]
	#acc_models_Wisconsin_n1 = [[],[]]
	#acc_models_Iris_n1 = [[],[]]
	#
	#for idx in range(len(Wine_model_cases_n1)):
	#	acc_models_Wine_n1[0].append(metrics.compute_acc_naive(Wine_model_cases_n1[idx].predict(X_Wine_train),y_Wine_train))
	#	acc_models_Wine_n1[1].append(metrics.compute_acc_naive(Wine_model_cases_n1[idx].predict(X_Wine_valid),y_Wine_valid))
	#	acc_models_Wisconsin_n1[0].append(metrics.compute_acc_naive(Wisconsin_model_cases_n1[idx].predict(X_Wisconsin_train),y_Wisconsin_train))
	#	acc_models_Wisconsin_n1[1].append(metrics.compute_acc_naive(Wisconsin_model_cases_n1[idx].predict(X_Wisconsin_valid),y_Wisconsin_valid))
	#	acc_models_Iris_n1[0].append(metrics.compute_acc_naive(Iris_model_cases_n1[idx].predict(X_Iris_train),y_Iris_train))
	#	acc_models_Iris_n1[1].append(metrics.compute_acc_naive(Iris_model_cases_n1[idx].predict(X_Iris_valid),y_Iris_Valid))

    #2 hidden layers

    #Wine_model_cases_n2 = load_models_params(X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid, 13, 3, regrasWine, RNGseed, nLayers = 2, debug = True)
    #Wisconsin_model_cases_n2 = load_models_params(X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid, 30, 2, regrasWisconsin, RNGseed, nLayers = 2, debug = True)
    #Iris_model_cases_n2 = load_models_params(X_Iris_train, X_Iris_valid, y_Iris_train,  y_Iris_valid, 4, 3, regrasIris, RNGseed, nLayers = 2, debug = True)

    #ruleSetsResults_2H_Wine_train = test_algorithms(Wine_model_cases_n2, X_Wine_train, Wine_classes, debug = True)
    #ruleSetsResults_2H_Wisconsin_train = test_algorithms(Wisconsin_model_cases_n2, X_Wisconsin_train, Wisconsin_classes, debug = True)
    #ruleSetsResults_2H_Iris_train = test_algorithms(Iris_model_cases_n2, X_Iris_train, Iris_classes, debug = True)

    #ruleSetsResults_2H_Wine_valid = test_algorithms(Wine_model_cases_n2, X_Wine_valid, Wine_classes, debug = True)
    #ruleSetsResults_2H_Wisconsin_valid = test_algorithms(Wisconsin_model_cases_n2, X_Wisconsin_valid, Wisconsin_classes, debug = True)
    #ruleSetsResults_2H_Iris_valid = test_algorithms(Iris_model_cases_n2, X_Iris_valid, Iris_classes, debug = True)

    #ruleSetsResults_2H_Wine = [ruleSetsResults_2H_Wine_train, ruleSetsResults_2H_Wine_valid]
    #ruleSetsResults_2H_Wisconsin = [ruleSetsResults_2H_Wisconsin_train, ruleSetsResults_2H_Wisconsin_valid]
    #ruleSetsResults_2H_Iris = [ruleSetsResults_2H_Iris_train, ruleSetsResults_2H_Iris_valid]

    #for idx in range(len(ruleSetResults_2H_Wine)):
    #   get predictions of ruleSets and the accuracy
    #   parseRulesTest(model, ruleSetsResults_2H_Wine, Wine_Database[0])
    #   computer_acc_rules_naive(ruleSetsResults_2H_Wine, Wine_Database, Wine_classes)
    #   computer_acc_rules_naive(ruleSetsResults_2H_Wisconsin, Wisconsin_Database, Wisconsin_classes)
    #   computer_acc_rules_naive(ruleSetsResults_2H_Iris, Iris_Database, Iris_classes)
	
	#acc_models_Wine_n2 = [[],[]]
	#acc_models_Wisconsin_n2 = [[],[]]
	#acc_models_Iris_n2 = [[],[]]
	#
	#for idx in range(len(Wine_model_cases_n2)):
	#	acc_models_Wine_n2[0].append(metrics.compute_acc_naive(Wine_model_cases_n2[idx].predict(X_Wine_train),y_Wine_train))
	#	acc_models_Wine_n2[1].append(metrics.compute_acc_naive(Wine_model_cases_n2[idx].predict(X_Wine_valid),y_Wine_valid))
	#	acc_models_Wisconsin_n2[0].append(metrics.compute_acc_naive(Wisconsin_model_cases_n2[idx].predict(X_Wisconsin_train),y_Wisconsin_train))
	#	acc_models_Wisconsin_n2[1].append(metrics.compute_acc_naive(Wisconsin_model_cases_n2[idx].predict(X_Wisconsin_valid),y_Wisconsin_valid))
	#	acc_models_Iris_n2[0].append(metrics.compute_acc_naive(Iris_model_cases_n2[idx].predict(X_Iris_train),y_Iris_train))
	#	acc_models_Iris_n2[1].append(metrics.compute_acc_naive(Iris_model_cases_n2[idx].predict(X_Iris_valid),y_Iris_Valid))

    #3 hidden layers

    #Wine_model_cases_n3 = load_models_params(X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid, 13, 3, regrasWine, RNGseed, nLayers = 3, debug = True)
    #Wisconsin_model_cases_n3 = load_models_params(X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid, 30, 2, regrasWisconsin, RNGseed, nLayers = 3, debug = True)
    #Iris_model_cases_n3 = load_models_params(X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid, 4, 3, regrasIris, RNGseed, nLayers = 3, debug = True)

    #ruleSetsResults_3H_Wine_train = test_algorithms(Wine_model_cases_n3, X_Wine_train, Wine_classes, debug = True)
    #ruleSetsResults_3H_Wisconsin_train = test_algorithms(Wisconsin_model_cases_n3, X_Wisconsin_train, Wisconsin_classes, debug = True)
    #ruleSetsResults_3H_Iris_train = test_algorithms(Iris_model_cases_n3, X_Iris_train, Iris_classes, debug = True)

    #ruleSetsResults_3H_Wine_valid = test_algorithms(Wine_model_cases_n3, X_Wine_valid, Wine_classes, debug = True)
    #ruleSetsResults_3H_Wisconsin_valid = test_algorithms(Wisconsin_model_cases_n3, X_Wisconsin_valid, Wisconsin_classes, debug = True)
    #ruleSetsResults_3H_Iris_valid = test_algorithms(Iris_model_cases_n3, X_Iris_valid, Iris_classes, debug = True)

    #ruleSetsResults_3H_Wine = [ruleSetsResults_3H_Wine_train, ruleSetsResults_3H_Wine_valid]
    #ruleSetsResults_3H_Wisconsin = [ruleSetsResults_3H_Wisconsin_train, ruleSetsResults_3H_Wisconsin_valid]
    #ruleSetsResults_3H_Iris = [ruleSetsResults_3H_Iris_train, ruleSetsResults_3H_Iris_valid]

    #for idx in range(len(ruleSetResults_3H_Wine)):
    #   get predictions of ruleSets and the accuracy
    #   parseRulesTest(model, ruleSetsResults_3H_Wine, Wine_Database[0])
    #   computer_acc_rules_naive(ruleSetsResults_3H_Wine, Wine_Database, Wine_classes)
    #   computer_acc_rules_naive(ruleSetsResults_3H_Wisconsin, Wisconsin_Database, Wisconsin_classes)
    #   computer_acc_rules_naive(ruleSetsResults_3H_Iris, Iris_Database, Iris_classes)
	
	#acc_models_Wine_n3 = [[],[]]
	#acc_models_Wisconsin_n3 = [[],[]]
	#acc_models_Iris_n3 = [[],[]]
	#
	#for idx in range(len(Wine_model_cases_n3)):
	#	acc_models_Wine_n3[0].append(metrics.compute_acc_naive(Wine_model_cases_n3[idx].predict(X_Wine_train),y_Wine_train))
	#	acc_models_Wine_n3[1].append(metrics.compute_acc_naive(Wine_model_cases_n3[idx].predict(X_Wine_valid),y_Wine_valid))
	#	acc_models_Wisconsin_n3[0].append(metrics.compute_acc_naive(Wisconsin_model_cases_n3[idx].predict(X_Wisconsin_train),y_Wisconsin_train))
	#	acc_models_Wisconsin_n3[1].append(metrics.compute_acc_naive(Wisconsin_model_cases_n3[idx].predict(X_Wisconsin_valid),y_Wisconsin_valid))
	#	acc_models_Iris_n3[0].append(metrics.compute_acc_naive(Iris_model_cases_n3[idx].predict(X_Iris_train),y_Iris_train))
	#	acc_models_Iris_n3[1].append(metrics.compute_acc_naive(Iris_model_cases_n3[idx].predict(X_Iris_valid),y_Iris_Valid))

    return

def simpleTest():
    seed = 1
    algoritmo_1_KT(seed)
    algoritmo_2_MofN(seed)
    algoritmo_3_RuleExtractLearning(seed)
    algoritmo_4_RxRen(seed)
    print("bateria de teste terminado")
    return

#simpleTest()
main_test()