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
from copy import deepcopy
import os
import gc

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
    ANN.train(train_X, train_y, valid_X, valid_y, epochs=5000, learning_rate=0.003)

    print(metrics.Compute_Acc_naive([ANN.predict(tX) for tX in train_X], train_y))
    print(metrics.Compute_Acc_naive([ANN.predict(vX) for vX in valid_X], valid_y))

    params = ANN.get_params()
    C = classes

    return ANN, C, [train_X,valid_X], [train_y, valid_y]

#TODO: fazer versão para parametros de modelo do SKLearn
def Neurons_to_Lists():
    pass

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
    result = KT.KT_1(U, C, debug=True)

    #KT.printRules(result)

    #print(metrics.Compute_Acc_naive([KT.parseRules(result, tX) for tX in DataX[0]], DataY[0]))
    #print(metrics.Compute_Acc_naive([KT.parseRules(result, vX) for vX in DataX[1]], DataY[1]))

    ANN.destroy()
    del ANN
    del C
    del DataX
    del DataY
    params.clear()
    del params
    for layer in U:
        for u in layer:
            del u
        del layer
    del U
    KT.delete(result)
    del result
    gc.collect()

def algoritmo_2_MofN(seed):
    ANN, _, DataX, Datay = load_example(seed)
    params = ANN.get_params()
    U = Neurons_to_Lists(params)
    result = MofN.MofN_2(U, ANN, DataX, Datay, debug=True)

    #MofN.printRules(result)

    #print(metrics.Compute_Acc_naive([MofN.parseRules(result, tX) for tX in DataX[0]], DataY[0]))
    #print(metrics.Compute_Acc_naive([MofN.parseRules(result, vX) for vX in DataX[1]], DataY[1]))

    ANN.destroy()
    del ANN
    del DataX
    del Datay
    params.clear()
    del params
    for layer in U:
        for u in layer:
            del u
        del layer
    del U
    MofN.delete(result)
    del result
    gc.collect()

def algoritmo_3_RuleExtractLearning(seed):
    ANN, C, DataX, DataY = load_example(seed)
    result = REL.Rule_extraction_learning_3(ANN, C, DataX[0], debug = True)

    #REL.printRules(result)

    #print(metrics.Compute_Acc_naive([REL.parseRules(result, tX) for tX in DataX[0]], DataY[0]))
    #print(metrics.Compute_Acc_naive([REL.parseRules(result, vX) for vX in DataX[1]], DataY[1]))

    ANN.destroy()
    del ANN
    del C
    del DataX
    REL.delete(result)
    del result
    gc.collect()

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
    #RxREN.printRules(resultado)

    #print(metrics.Compute_Acc_naive([RxREN.parseRules(resultado, tX) for tX in DataX[0]], Datay[0]))
    #print(metrics.Compute_Acc_naive([RxREN.parseRules(resultado, vX) for vX in DataX[1]], Datay[1]))

    ANN.destroy()
    del ANN
    del C
    del DataX
    del Datay
    params.clear()
    del params
    for layer in U:
        for u in layer:
            del u
        del layer
    del U
    RxREN.delete(resultado)
    del resultado
    gc.collect()

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
        model = case[0]
        correct_cases = case[1]

        tempoInicio = time.time()
        algo1_result = KT.KT_1(Neurons_to_Lists(deepcopy(model.get_params())), classes, debug = debug)
        tempoCheckpoint1 = time.time()
        algo2_result = MofN.MofN_2(Neurons_to_Lists(deepcopy(model.get_params())), deepcopy(model), dataBase[0], dataBase[1], debug = debug)
        tempoCheckpoint2 = time.time()
        algo3_result = None#REL.Rule_extraction_learning_3(deepcopy(model), classes, dataBase[0][1], debug = debug)
        tempoCheckpoint3 = time.time()
        algo4_result = RxREN.RxREN_4(model, Neurons_to_Lists(deepcopy(model.get_params())), correct_cases[0], correct_cases[1], classes, debug = debug)
        tempoCheckpoint4 = time.time()

        result = [[algo1_result, algo2_result, algo3_result, algo4_result], [tempoCheckpoint1 - tempoInicio, tempoCheckpoint2 - tempoCheckpoint1, tempoCheckpoint3 - tempoCheckpoint2, tempoCheckpoint4 - tempoCheckpoint3]]
        results.append(result)

    return results

def completeness_check(ruleSets):
    KT_check = KT.isComplete(ruleSets[0])
    MofN_check = MofN.isComplete(ruleSets[1])
    REL_check = REL.isComplete(ruleSets[2])
    RxREN_check = RxREN.isComplete(ruleSets[3])
    return [KT_check, MofN_check, REL_check, RxREN_check]

def parseRulesTest(model, ruleSets, X):
    pred_results = []

    for x_set in X:
        set_results = []
        for x_case in x_set:
            KT_result = KT.parseRules(ruleSets[0], model, x_case) #if KT.isComplete(ruleSets[0]) else "Error"
            MofN_result = MofN.parseRules(ruleSets[1], model, x_case) #if MofN.isComplete(ruleSets[1]) else "Error"
            REL_result = REL.parseRules(ruleSets[2], x_case) if REL.isComplete(ruleSets[2]) else "Error"
            RxREN_result = RxREN.parseRules(ruleSets[3], x_case) #if RxREN.isComplete(ruleSets[3]) else "Error"

            set_results.append([KT_result, MofN_result, REL_result, RxREN_result])

        pred_results.append(set_results)

    return pred_results

def classArrayConvertion(preds, classes):
    preds01 = []
    for predCase in preds:
        print(classes)
        print(predCase)
        if isinstance(predCase, tuple):
            preds01([int(idx == predCase[1]) for idx, classVal in classes])
        elif isinstance(predCase, list):
            preds01.append([int(classVal in predCase) for classVal in classes])
        else:
            preds01.append([int(classVal in {predCase}) for classVal in classes])

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

def testesBateria(Database, Classes, numHLayers, HLayerTam, entrada, saida, RNGseed, nomeDatabase, debug = False):
    #Database = [[X_train, X_valid],[y_train, y_valid]]
    modelCases = load_models_params(Database[0][0], Database[0][1], Database[1][0], Database[1][1], entrada, saida, RNGseed, [HLayerTam], nLayers = numHLayers, debug = True)
    modelCasesAcc = [[model[0].accuracy(Database[0][0], Database[1][0]) for model in modelCases],[model[0].accuracy(Database[0][1], Database[1][1]) for model in modelCases]]
    print("models made")
    if debug:
        print(modelCases)
        print(modelCasesAcc)

    ruleSetsResults = test_algorithms(modelCases, Database, Classes, debug = debug)
    print("rules made")
    missing_entries = []
    for idx, ruleSetCase in enumerate(ruleSetsResults):
        setEvaluation = ruleSetCase[0]
        HiddenLayerRule = HLayerTam[idx] if isinstance(HLayerTam, list) else HLayerTam
        if not KT.isComplete(setEvaluation[0]):
            numberRulesLayeredKT = []
            for layer_r in setEvaluation[0]:
                numberRulesLayeredKT.append(len(layer_r))

            missing_entries.append([nomeDatabase, entrada, saida, numHLayers, HiddenLayerRule, "KT", numberRulesLayeredKT])

        if not MofN.isComplete(setEvaluation[1]):
            numberRulesLayeredMofN = []
            for layer_r in setEvaluation[1]:
                numberRulesLayeredMofN.append(len(layer_r))
            missing_entries.append([nomeDatabase, entrada, saida, numHLayers, HiddenLayerRule, "MofN", numberRulesLayeredMofN])

        if not REL.isComplete(setEvaluation[2]):
            numberRulesLayeredREL = dict()
            if setEvaluation[2] is not None:
                for classResult, rules in setEvaluation[2].items():
                    numberRulesLayeredREL[classResult] = len(rules)
            missing_entries.append([nomeDatabase, entrada, saida, numHLayers, HiddenLayerRule, "REL", numberRulesLayeredREL])

        if not RxREN.isComplete(setEvaluation[3]):
            numberRulesLayeredRxREN = dict()
            for classResult, rules in setEvaluation[3].items():
                numberRulesLayeredRxREN[classResult] = len(rules)
            missing_entries.append([nomeDatabase, entrada, saida, numHLayers, HiddenLayerRule, "RxREN", numberRulesLayeredRxREN])

    print_missing_entries(missing_entries)
    print("missing entries checked")

    rulePred = [parseRulesTest(model[0], ruleSet[0], Database[0]) for model, ruleSet in zip(modelCases, ruleSetsResults)]
    ruleAcc = [compute_acc_rules_naive(pred, Database, Classes) for pred in rulePred]
    ExecuteTime = [ruleSet[1] for ruleSet in ruleSetsResults]
    print("evaluation done")

    for model in modelCases:
        print("apagando", model[0])
        model[0].destroy()
        del model[0]
        del model
    del modelCases
    del missing_entries
    for results in ruleSetsResults:
        KT.delete(results[0][0])
        MofN.delete(results[0][1])
        REL.delete(results[0][2])
        RxREN.delete(results[0][3])
        del results
    del ruleSetsResults
    del rulePred
    gc.collect()
    print("teste do modelo: database " + nomeDatabase + " com " + str(numHLayers) + " camadas ocultas com tamanho " + str(HLayerTam))
    return [modelCasesAcc, ruleAcc, ExecuteTime]

def main_test(RNGseed):
    np.random.seed(RNGseed)

    if not os.path.exists("resultados"):
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
    decisionTree_Wine = DecisionTreeClassifier(max_depth = 3, random_state = RNGseed)
    decisionTree_Wine.fit(X_Wine_train, y_Wine_train)
    acc_decisionTree_Wine_train = metrics.Compute_Acc_naive(decisionTree_Wine.predict(X_Wine_train), y_Wine_train)
    acc_decisionTree_Wine_valid = metrics.Compute_Acc_naive(decisionTree_Wine.predict(X_Wine_valid), y_Wine_valid)

    decisionTree_Wisconsin = DecisionTreeClassifier(max_depth = 3, random_state = RNGseed)
    decisionTree_Wisconsin.fit(X_Wisconsin_train, y_Wisconsin_train)
    acc_decisionTree_Wisconsin_train = metrics.Compute_Acc_naive(decisionTree_Wisconsin.predict(X_Wisconsin_train), y_Wisconsin_train)
    acc_decisionTree_Wisconsin_valid = metrics.Compute_Acc_naive(decisionTree_Wisconsin.predict(X_Wisconsin_valid), y_Wisconsin_valid)

    decisionTree_Iris = DecisionTreeClassifier(max_depth = 3, random_state = RNGseed)
    decisionTree_Iris.fit(X_Iris_train, y_Iris_train)
    acc_decisionTree_Iris_train = metrics.Compute_Acc_naive(decisionTree_Iris.predict(X_Iris_train), y_Iris_train)
    acc_decisionTree_Iris_valid = metrics.Compute_Acc_naive(decisionTree_Iris.predict(X_Iris_valid), y_Iris_valid)

    enunciadoLinha = ["Algoritmo"].extend(regras)
    with open('resultados/resultados_arvores.csv', 'a+', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Database","Set","acuracy"])
        writer.writerow(["Wine", "train", acc_decisionTree_Wine_train])
        writer.writerow(["Wine", "valid", acc_decisionTree_Wine_valid])
        writer.writerow(["Wisconsin", "train", acc_decisionTree_Wisconsin_train])
        writer.writerow(["Wisconsin", "valid", acc_decisionTree_Wisconsin_valid])
        writer.writerow(["Iris", "train", acc_decisionTree_Iris_train])
        writer.writerow(["Iris", "valid", acc_decisionTree_Iris_valid])

    del acc_decisionTree_Wine_train
    del acc_decisionTree_Wine_valid
    del acc_decisionTree_Wisconsin_train
    del acc_decisionTree_Wisconsin_valid
    del acc_decisionTree_Iris_train
    del acc_decisionTree_Iris_valid
    gc.collect()

    #"E", "E + 1", "2E - 1", "2E", "S", "S + 1", "2S - 1", "2S", "(E + S)/2", "(2E + S)/3"
    #"E", "S", "(E + S)/2", "(2E + S)/3"

    WineEntrada = 13
    WineSaida = 3
    WineHiddenLayerLen = [WineEntrada, WineEntrada + 1, 2*WineEntrada - 1, 2*WineEntrada, WineSaida, WineSaida + 1, 2*WineSaida - 1, 2*WineSaida, math.ceil((WineSaida + WineEntrada)/2), math.ceil((2*WineEntrada + WineSaida)/3)]
    WineHiddenLayerLenShort = [WineEntrada, WineSaida, math.ceil((WineSaida + WineEntrada)/2), math.ceil((2*WineEntrada + WineSaida)/3)]
    WineHiddenLayerLenShort1 = [WineEntrada + 1, 2*WineEntrada - 1, 2*WineEntrada, WineSaida + 1, 2*WineSaida - 1, 2*WineSaida]

    WisconsinEntrada = 30
    WisconsinSaida = 2
    WisconsinHiddenLayerLen = [WisconsinEntrada, WisconsinEntrada + 1, 2*WisconsinEntrada - 1, 2*WisconsinEntrada, WisconsinSaida, WisconsinSaida + 1, 2*WisconsinSaida - 1, 2*WisconsinSaida, math.ceil((WisconsinSaida + WisconsinEntrada)/2), math.ceil((2*WisconsinEntrada + WisconsinSaida)/3)]
    WisconsinHiddenLayerLenShort = [WisconsinEntrada, WisconsinSaida, math.ceil((WisconsinSaida + WisconsinEntrada)/2), math.ceil((2*WisconsinEntrada + WisconsinSaida)/3)]
    WisconsinHiddenLayerLenShort1 = [WisconsinEntrada + 1, 2*WisconsinEntrada - 1, 2*WisconsinEntrada, WisconsinSaida + 1, 2*WisconsinSaida - 1, 2*WisconsinSaida]

    IrisEntrada = 4
    IrisSaida = 3
    IrisHiddenLayerLen = [IrisEntrada, IrisEntrada + 1, 2*IrisEntrada - 1, 2*IrisEntrada, IrisSaida, IrisSaida + 1, 2*IrisSaida - 1, 2*IrisSaida, math.ceil((IrisSaida + IrisEntrada)/2), math.ceil((2*IrisEntrada + IrisSaida)/3)]
    IrisHiddenLayerLenShort = [IrisEntrada, IrisSaida, math.ceil((IrisSaida + IrisEntrada)/2), math.ceil((2*IrisEntrada + IrisSaida)/3)]
    IrisHiddenLayerLenShort1 = [IrisEntrada + 1, 2*IrisEntrada - 1, 2*IrisEntrada, IrisSaida + 1, 2*IrisSaida - 1, 2*IrisSaida]

    #1 hidden layer

    Wine_1H_AccModelRule = [testesBateria(Wine_Database, Wine_classes, 1, HLayerTam, 13, 3, RNGseed, "Wine", debug = True) for HLayerTam in WineHiddenLayerLenShort]
    print_Test_results(Wine_1H_AccModelRule, 'resultados/resultados_tests_1H.csv', "Wine")

    #Wisconsin_1H_AccModelRule = [testesBateria(Wisconsin_Database, Wisconsin_classes, 1, HLayerTam, 30, 2, RNGseed, "Wisconsin", debug = True) for HLayerTam in WisconsinHiddenLayerLenShort]
    #print_Test_results(Wisconsin_1H_AccModelRule, 'resultados/resultados_tests_1H.csv', "Wisconsin")

    Iris_1H_AccModelRule = [testesBateria(Iris_Database, Iris_classes, 1, HLayerTam, 4, 3, RNGseed, "Iris", debug = True) for HLayerTam in IrisHiddenLayerLenShort]
    print_Test_results(Iris_1H_AccModelRule, 'resultados/resultados_tests_1H.csv', "Iris")

    #2 hidden layers

    Wine_2H_AccModelRule = [testesBateria(Wine_Database, Wine_classes, 2, HLayerTam, 13, 3, RNGseed, "Wine", debug = True) for HLayerTam in WineHiddenLayerLenShort]
    print_Test_results(Wine_2H_AccModelRule, 'resultados/resultados_tests_2H.csv', "Wine")

    #Wisconsin_2H_AccModelRule = [testesBateria(Wisconsin_Database, Wisconsin_classes, 2, HLayerTam, 30, 2, RNGseed, "Wisconsin", debug = True) for HLayerTam in WisconsinHiddenLayerLenShort]
    #print_Test_results(Wisconsin_2H_AccModelRule, 'resultados/resultados_tests_2H.csv', "Wisconsin")

    Iris_2H_AccModelRule = [testesBateria(Iris_Database, Iris_classes, 2, HLayerTam, 4, 3, RNGseed, "Iris", debug = True) for HLayerTam in IrisHiddenLayerLenShort]
    print_Test_results(Iris_2H_AccModelRule, 'resultados/resultados_tests_2H.csv', "Iris")

    #3 hidden layers

    Wine_3H_AccModelRule = [testesBateria(Wine_Database, Wine_classes, 3, HLayerTam, 13, 3, RNGseed, "Wine", debug = True) for HLayerTam in WineHiddenLayerLenShort]
    print_Test_results(Wine_3H_AccModelRule, 'resultados/resultados_tests_3H.csv', "Wine")

    #Wisconsin_3H_AccModelRule = [testesBateria(Wisconsin_Database, Wisconsin_classes, 3, HLayerTam, 30, 2, RNGseed, "Wisconsin", debug = True) for HLayerTam in WisconsinHiddenLayerLenShort]
    #print_Test_results(Wisconsin_3H_AccModelRule, 'resultados/resultados_tests_3H.csv', "Wisconsin")

    Iris_3H_AccModelRule = [testesBateria(Iris_Database, Iris_classes, 3, HLayerTam, 4, 3, RNGseed, "Iris", debug = True) for HLayerTam in IrisHiddenLayerLenShort]
    print_Test_results(Iris_3H_AccModelRule, 'resultados/resultados_tests_3H.csv', "Iris")

    print("bateria de teste principal terminado")
    del WineEntrada
    del WineSaida
    del WineHiddenLayerLen

    del WisconsinEntrada
    del WisconsinSaida
    del WisconsinHiddenLayerLen

    del IrisEntrada
    del IrisSaida
    del IrisHiddenLayerLen

    gc.collect()

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

def print_Test_results(resultArray, fileName, DataBaseName):
    #To fix later: the result array is nesting weirdly, but I need to finish this projet ASAP
    with open(fileName, 'a+', newline= '', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["DataBase", "E", "S", "(E + S)/2", "(2E + S)/3"])
        trainRow = [DataBaseName + "_train"] + [resultItem[0][0] for resultItem in resultArray]
        validRow = [DataBaseName + "_valid"] + [resultItem[0][1] for resultItem in resultArray]
        writer.writerow(trainRow)
        writer.writerow(validRow)

        writer.writerow(["Train"])
        writer.writerow(["Algoritmo", "E", "S", "(E + S)/2", "(2E + S)/3"])
        KTRow = ["KT"] + [resultItem[1][0][0][0] for resultItem in resultArray]
        MofNRow = ["MofN"] + [resultItem[1][0][0][1] for resultItem in resultArray]
        RELRow = ["REL"] + [resultItem[1][0][0][2] for resultItem in resultArray]
        RxRENRow = ["RxREN"] + [resultItem[1][0][0][3] for resultItem in resultArray]
        writer.writerow(KTRow)
        writer.writerow(MofNRow)
        writer.writerow(RELRow)
        writer.writerow(RxRENRow)

        writer.writerow(["Valid"])
        writer.writerow(["Algoritmo", "E", "S", "(E + S)/2", "(2E + S)/3"])
        KTRow = ["KT"] + [resultItem[1][0][1][0] for resultItem in resultArray]
        MofNRow = ["MofN"] + [resultItem[1][0][1][1] for resultItem in resultArray]
        RELRow = ["REL"] + [resultItem[1][0][1][2] for resultItem in resultArray]
        RxRENRow = ["RxREN"] + [resultItem[1][0][1][3] for resultItem in resultArray]
        writer.writerow(KTRow)
        writer.writerow(MofNRow)
        writer.writerow(RELRow)
        writer.writerow(RxRENRow)

        writer.writerow(["time of execution (seconds)"])
        KTRow = ["KT"] + [resultItem[2][0][0] for resultItem in resultArray]
        MofNRow = ["MofN"] + [resultItem[2][0][1] for resultItem in resultArray]
        RELRow = ["REL"] + [resultItem[2][0][2] for resultItem in resultArray]
        RxRENRow = ["RxREN"] + [resultItem[2][0][3] for resultItem in resultArray]
        writer.writerow(KTRow)
        writer.writerow(MofNRow)
        writer.writerow(RELRow)
        writer.writerow(RxRENRow)
        writer.writerow([])

def simpleTest(seed):
    algoritmo_1_KT(seed)
    #algoritmo_2_MofN(seed)
    #algoritmo_3_RuleExtractLearning(seed)
    #algoritmo_4_RxRen(seed)
    print("sem erros executando")

    return

#seed = random.randrange(4294967296)
seed = 0
#simpleTest(seed)
main_test(seed)
print(seed)