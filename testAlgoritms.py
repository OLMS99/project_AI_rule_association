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
    for case in modelParamsList:
        model = case[0].copy()
        correct_cases = case[1]

        if debug: 
            print(Neurons_to_Lists(model.get_params()))

        tempoInicio = time.time()
        algo1_result = KT.KT_1(Neurons_to_Lists(model.get_params()))
        tempoCheckpoint1 = time.time()
        algo2_result = MofN.MofN_2(Neurons_to_Lists(model.get_params()), model, dataBase[0], dataBase[1])
        tempoCheckpoint2 = time.time()
        algo3_result = REL.Rule_extraction_learning_3(model, classes, dataBase[0][1])
        tempoCheckpoint3 = time.time()
        algo4_result = RxREN.RxREN_4(model, Neurons_to_Lists(model.get_params()), correct_cases[0], correct_cases[1], classes)
        tempoCheckpoint4 = time.time()

    results.append([[algo1_result, algo2_result, algo3_result, algo4_result], [tempoCheckpoint1 - tempoInicio, tempoCheckpoint2 - tempoCheckpoint1, tempoCheckpoint3 - tempoCheckpoint2, tempoCheckpoint4 - tempoCheckpoint4]])
    return results

def parseRulesTest(model, ruleSets, X):
    pred_results = []

    for x_set in X:
        set_results = []
        for x_case in x_set:
            print(ruleSets)
            KT_result = KT.parseRules(ruleSets[0], model, x_case) if KT.isComplete(ruleSets[0]) else "Error"
            MofN_result = MofN.parseRules(ruleSets[1], model, x_case) if MofN.isComplete(ruleSets[1]) else "Error"
            REL_result = REL.parseRules(ruleSets[2], x_case) if REL.isComplete(ruleSets[2]) else "Error"
            RxREN_result = RxREN.parseRules(ruleSets[3], x_case) if RxREN.isComplete(ruleSets[3]) else "Error"

            set_results.append([KT_result, MofN_result, REL_result, RxREN_result])

        pred_results.append(set_results)

    return pred_results

def classArrayConvertion(preds, classes):
    preds01 = []
    for predCase in preds:
        if predCase == "Error" or predCase == float('inf') or predCase[0] == "no_results" or predCase[0] == "no_output_value":
            preds01.append([0 for classVal in classes])
        else:
            preds01.append([int(classVal in predCase) for classVal in classes])

    return preds01

def compute_acc_rules_naive(ruleResults, Database, classes):
    #Database = [[X_train, X_valid],[y_train, y_valid]]
    Acc_results = []
    for idx, ruleResultsSet in enumerate(ruleResults):
        #Converte a lista para preds de cada algoritmos
        KT_preds = [result[0] for result in ruleResultsSet]
        MofN_preds = [result[1] for result in ruleResultsSet]
        REL_preds = [result[2] for result in ruleResultsSet]
        RxREN_preds = [result[3] for result in ruleResultsSet]

        #Converte a predições para listas de 0 e 1
        KT_preds = classArrayConvertion(KT_preds, classes)
        MofN_preds = classArrayConvertion(MofN_preds, classes)
        REL_preds = classArrayConvertion(REL_preds, classes)
        RxREN_preds = classArrayConvertion(RxREN_preds, classes)

        y_set = Database[1][idx]

        #calular a accuracia com o acc naive [flexibilizar depois]
        KT_acc = metrics.Compute_Acc_naive(KT_preds, y_set)
        MofN_acc = metrics.Compute_Acc_naive(MofN_preds, y_set)
        REL_acc = metrics.Compute_Acc_naive(REL_preds, y_set)
        RxREN_acc = metrics.Compute_Acc_naive(RxREN_preds, y_set)

        Acc_results.append([KT_acc, MofN_acc, REL_acc, RxREN_acc])

    return Acc_results

def testesBateria(Database, Classes, numHLayers, entrada, saida, RNGseed, nomeDatabase, debug = False):
    #Database = [[X_train, X_valid],[y_train, y_valid]]
    regras = [entrada, entrada + 1, 2*entrada - 1, 2*entrada, saida, saida + 1, 2*saida - 1, 2*saida, math.ceil((saida + entrada)/2), math.ceil((2*entrada + saida)/3)]
    modelCases = load_models_params(Database[0][0], Database[0][1], Database[1][0], Database[1][1], entrada, saida, regras, RNGseed, debug = True)
    modelCasesAcc = [[model[0].accuracy(Database[0][0], Database[1][0]) for model in modelCases],[model[0].accuracy(Database[0][1], Database[1][1]) for model in modelCases]]

    if debug:
        print(modelCases)
        print(modelCasesAcc)

    ruleSetsResults = test_algorithms(modelCases, Database, Classes, debug = debug)

    missing_entries = []
    for idx, ruleSetCase in enumerate(ruleSetsResults):
        setEvaluation = ruleSetCase[0]

        if not KT.isComplete(setEvaluation[0]):
            missing_entries.append([nomeDatabase, entrada, saida, numHLayers, regras[idx], "KT", setEvaluation[0]])

        if not MofN.isComplete(setEvaluation[1]):
            missing_entries.append([nomeDatabase, entrada, saida, numHLayers, regras[idx], "MofN", setEvaluation[1]])

        if not REL.isComplete(setEvaluation[2]):
            missing_entries.append([nomeDatabase, entrada, saida, numHLayers, regras[idx], "REL", setEvaluation[2]])

        if not RxREN.isComplete(setEvaluation[3]):
            missing_entries.append([nomeDatabase, entrada, saida, numHLayers, regras[idx], "RxREN", setEvaluation[3]])

    print_missing_entries(missing_entries)

    rulePred = [parseRulesTest(model[0], ruleSet[0], Database[0]) for model, ruleSet in zip(modelCases, ruleSetsResults)]
    ruleAcc = [compute_acc_rules_naive(pred, Database, Classes) for pred in rulePred]
    ExecuteTime = [ruleSet[1] for ruleSet in ruleSetsResults]

    return [modelCasesAcc, ruleAcc, ExecuteTime]

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

    lista_Sem_Regras_Iris = []
    lista_Sem_Regras_Wine = []
    lista_Sem_Regras_Wiscosin = []

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

    enunciadoLinha = ["Algoritmo"].extend(regras)
    with open('resultados/resultados_arvores.csv', 'w', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Database","Set","acuracy"])
        writer.writerow(["Wine", "train", acc_decisionTree_Wine_train])
        writer.writerow(["Wine", "valid", acc_decisionTree_Wine_valid])
        writer.writerow(["Wisconsin", "train", acc_decisionTree_Wisconsin_train])
        writer.writerow(["Wisconsin", "valid", acc_decisionTree_Wisconsin_valid])
        writer.writerow(["Iris", "train", acc_decisionTree_Iris_train])
        writer.writerow(["Iris", "valid", acc_decisionTree_Iris_valid])

    WineEntrada = 13
    WineSaida = 3
    WineHiddenLayerLen = [WineEntrada, WineEntrada + 1, 2*WineEntrada - 1, 2*WineEntrada, WineSaida, WineSaida + 1, 2*WineSaida - 1, 2*WineSaida, math.ceil((WineSaida + WineEntrada)/2), math.ceil((2*WineEntrada + WineSaida)/3)]

    WisconsinEntrada = 13
    WisconsinSaida = 3
    WisconsinHiddenLayerLen = [WisconsinEntrada, WisconsinEntrada + 1, 2*WisconsinEntrada - 1, 2*WisconsinEntrada, WisconsinSaida, WisconsinSaida + 1, 2*WisconsinSaida - 1, 2*WisconsinSaida, math.ceil((WisconsinSaida + WisconsinEntrada)/2), math.ceil((2*WisconsinEntrada + WisconsinSaida)/3)]

    IrisEntrada = 13
    IrisSaida = 3
    WineHiddenLayerLen = [IrisEntrada, IrisEntrada + 1, 2*IrisEntrada - 1, 2*IrisEntrada, IrisSaida, IrisSaida + 1, 2*IrisSaida - 1, 2*IrisSaida, math.ceil((IrisSaida + IrisEntrada)/2), math.ceil((2*IrisEntrada + IrisSaida)/3)]

    #1 hidden layer

    Wine_1H_AccModelRule = testesBateria(Wine_Database, Wine_classes, 1, 13, 3, RNGseed, "Wine", debug = False)
    #Wisconsin_1H_AccModelRule = testesBateria(Wisconsin_Database, Wisconsin_classes, 1, 30, 2, RNGseed, "Wisconsin", debug = True)
    Iris_1H_AccModelRule = testesBateria(Iris_Database, Iris_classes, 1, 4, 3, RNGseed, "Iris", debug = False)

    with open('resultados/resultados_tests_1H.csv', 'w', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Wine_model_train","Wine_model_valid"])
        writer.writerow(Wine_1H_AccModelRule[0])
        writer.writerow(enunciadoLinha)
        #writer.writerow(["KT"].extend([accuracyCase[0] for accuracyCase in Wine_1H_AccModelRule[1][0]]))
        #writer.writerow(["MofN"].extend([accuracyCase[1] for accuracyCase in Wine_1H_AccModelRule[1][0]]))
        #writer.writerow(["REL"].extend([accuracyCase[2] for accuracyCase in Wine_1H_AccModelRule[1][0]]))
        #writer.writerow(["RxREN"].extend([accuracyCase[3] for accuracyCase in Wine_1H_AccModelRule[1][0]]))
        writer.writerow(Wine_1H_AccModelRule[1][0])
        writer.writerow("time of execution")
        #writer.writerow(["KT"].extend([accuracyCase[0] for accuracyCase in Wine_1H_AccModelRule[2][0]]))
        #writer.writerow(["MofN"].extend([accuracyCase[1] for accuracyCase in Wine_1H_AccModelRule[2][0]]))
        #writer.writerow(["REL"].extend([accuracyCase[2] for accuracyCase in Wine_1H_AccModelRule[2][0]]))
        #writer.writerow(["RxREN"].extend([accuracyCase[3] for accuracyCase in Wine_1H_AccModelRule[2][0]]))
        writer.writerow(Wine_1H_AccModelRule[2][0])
        #writer.writerow(["Wisconsin_model_train","Wisconsin_model_valid"])
        #writer.writerow(Wisconsin_1H_AccModelRule[0])
        #writer.writerow(enunciadoLinha)
        #writer.writerow(Wisconsin_1H_AccModelRule[1][0][0])
        #writer.writerow(Wisconsin_1H_AccModelRule[1][0][1])
        #writer.writerow(Wisconsin_1H_AccModelRule[1])
        writer.writerow(["Iris_model_train","Iris_model_valid"])
        writer.writerow(Iris_1H_AccModelRule[0])
        writer.writerow(enunciadoLinha)
        writer.writerow(Iris_1H_AccModelRule[1][0][0])
        writer.writerow(Iris_1H_AccModelRule[1][0][1])
        writer.writerow(Iris_1H_AccModelRule[1])

    #2 hidden layers

    Wine_2H_AccModelRule = testesBateria(Wine_Database, Wine_classes, 2, 13, 3, RNGseed, "Wine", debug = False)
    #Wisconsin_2H_AccModelRule = testesBateria(Wisconsin_Database, Wisconsin_classes, 2, 30, 2, RNGseed, "Wisconsin", debug = True)
    Iris_2H_AccModelRule = testesBateria(Iris_Database, Iris_classes, 2, 4, 3, RNGseed, "Iris", debug = False)

    with open('resultados/resultados_tests_2H.csv', 'w', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Wine_model_train","Wine_model_valid"])
        writer.writerow(Wine_2H_AccModelRule[0])
        writer.writerow(enunciadoLinha)
        writer.writerow(Wine_2H_AccModelRule[1])
        writer.writerow(Wine_2H_AccModelRule[2])
        #writer.writerow(["Wisconsin_model_train","Wisconsin_model_valid"])
        #writer.writerow(Wisconsin_2H_AccModelRule[0])
        #writer.writerow(enunciadoLinha)
        #writer.writerow(Wisconsin_2H_AccModelRule[1][0][0])
        #writer.writerow(Wisconsin_2H_AccModelRule[1][0][1])
        #writer.writerow(Wisconsin_2H_AccModelRule[1])
        writer.writerow(["Iris_model_train","Iris_model_valid"])
        writer.writerow(Iris_2H_AccModelRule[0])
        writer.writerow(enunciadoLinha)
        writer.writerow(Iris_2H_AccModelRule[1][0][0])
        writer.writerow(Iris_2H_AccModelRule[1][0][1])
        writer.writerow(Iris_2H_AccModelRule[1])

    #3 hidden layers

    Wine_3H_AccModelRule = testesBateria(Wine_Database, Wine_classes, 3, 13, 3, RNGseed, "Wine", debug = False)
    #Wisconsin_3H_AccModelRule = testesBateria(Wisconsin_Database, Wisconsin_classes, 3, 30, 2, RNGseed, "Wisconsin", debug = True)
    Iris_3H_AccModelRule = testesBateria(Iris_Database, Iris_classes, 3, 4, 3, RNGseed, "Iris", debug = False)


    with open('resultados/resultados_tests_3H.csv', 'w', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Wine_model_train","Wine_model_valid"])
        writer.writerow(Wine_3H_AccModelRule[0])
        writer.writerow(enunciadoLinha)
        writer.writerow(Wine_3H_AccModelRule[1])
        writer.writerow(Wine_3H_AccModelRule[2])

        #writer.writerow(["Wisconsin_model_train","Wisconsin_model_valid"])
        #writer.writerow(Wisconsin_3H_AccModelRule[0])
        #writer.writerow(enunciadoLinha)
        #writer.writerow(Wisconsin_3H_AccModelRule[1][0][0])
        #writer.writerow(Wisconsin_3H_AccModelRule[1][0][1])
        #writer.writerow(Wisconsin_3H_AccModelRule[1])
        writer.writerow(["Iris_model_train","Iris_model_valid"])
        writer.writerow(Iris_3H_AccModelRule[0])
        writer.writerow(enunciadoLinha)
        writer.writerow(Iris_3H_AccModelRule[1][0][0])
        writer.writerow(Iris_3H_AccModelRule[1][0][1])
        writer.writerow(Iris_3H_AccModelRule[1])

    print("###################################################################################################")
    print(Wine_1H_AccModelRule)
    print(Iris_1H_AccModelRule)
    print("###################################################################################################")
    print(Wine_2H_AccModelRule)
    print(Iris_2H_AccModelRule)
    print("###################################################################################################")
    print(Wine_3H_AccModelRule)
    print(Iris_3H_AccModelRule)
    print("bateria de teste principal terminado")
    return

def print_missing_entries(listaCasos):
    if not os.path.exists("log"):
        os.mkdir("log")

    if not os.path.exists("log/casos_sem_regras.csv"):
        with open('log/casos_sem_regras.csv', 'w', newline= '', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Database", "entrada","saida","numero de camadas ocultas","tamanho camada oculta","algoritmo","visao do resultado"])

    with open('log/casos_sem_regras.csv', 'a+', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(listaCasos)

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