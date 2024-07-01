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

    train_X, valid_X, train_y, valid_y = train_test_split(data, label_target, test_size = split_test_size, random_state = 13)

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
    ANN, C, DataX, DataY = load_example()
    params = ANN.get_params()
    U = Neurons_to_Lists(params)
    result = KT.KT_1(U, debug=True)

    if len(result) > 0:
        for r in result:
            r.print()
    else:
        print("nenhuma regra feita")

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

    #for case in DataX[0]:
        #tenta todas as arvores e ve os resultados
        #resposta = result.step(case)
        #compare


def algoritmo_3_RuleExtractLearning():
    ANN, C, DataX, _ = load_example()
    result = REL.Rule_extraction_learning_3(ANN, C, DataX[0], debug = True)
    for r in result.keys():
        if result[r]:
            print("rule made for %s" % (r))
            result[r].print()
        else:
            print("no rule made for %s" % (r))

    for case in DataX[0]:
        #tenta todas as arvores e ve qual da true e qual da false
        resposta = resultado.step(case)
        #compare

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

    for case in T:
        resposta = resultado.step(case)
        #compare

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

def load_models_params(x_train, x_valid, y_train, y_valid, nEntrada, nSaida, nLayers = 1):
    regras=[nEntrada, nEntrada+1, 2*nEntrada-1, 2*nEntrada, nSaida, nSaida+1, 2*nSaida-1, 2*nSaida, (nEntrada+nSaida)/2,(nEntrada*2/3+nSaida)]
    results = []
    for caso in regras:

        ordem = []
        ordem.append(nEntrada)
        for i in range(nLayers):
            ordem.append(caso)
        ordem.append(nSaida)

        model = NN.nnf(ordem, [ACT.sigmoid]*(nLayers+2), Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
        model.train(x_train, y_train, x_valid, y_valid, epochs=1000, learning_rate=0.01)

        params = model.get_params()
        results.append([model, params])

    return results

def test_algorithms(classes, x_train, y_train, x_valid, y_valid, params, model):

    U = Neurons_to_Lists(params)
    result_KT = KT.KT_1(U)

    result_MofN = MofN.Mof_2(model, [x_train, x_valid],[y_train, y_valid])

    exemplos_classificados_corretamente = filter_correct_answers()
    result_REL = REL.Rule_extraction_learning_3(model, classes, x_train)

    result_RxREN = 0

    return

def step_kt(conjuntoRegras, conjuntoInput):
    conjuntoRegrasLocal = np.array(conjuntoRegras)
    conjuntoInputLocal = np.array(conjuntoInput)
    lastLayer = 0
    for regra in conjuntoRegrasLocal:
        lastLayer = max(lastLayer, regra.getInputNeuron()[0]+1)

    RegrasDivididas = []
    for layer in range(lastLayer):
        RegrasCamada = [conjuntoRegrasLocal[i] for i in range(len(conjuntoRegrasLocal)) if conjuntoRegrasLocal[i].getInputNeuron()[0] == layer]
        RegrasDivididas.append(RegrasCamada)

    ConjuntoResultado = None
    for layer in regrasDivididas:
        ConjuntoResultado = [i.step(conjuntoInputLocal) for i in layer]
        conjuntoInputLocal = [ConjuntoResultado[i] for i in range(len(ConjuntoResultado))]

    return ConjuntoResultado

def main_test():
    decisionTreeSeed = 42
    seed = 1
    np.random.seed(seed)

    #carregar exemplos e classes dos exemplos
    Iris_classes, X_Iris_train, X_Iris_valid, y_Iris_train, y_Iris_valid = load_iris_cobaia(seed)
    Wine_classes, X_Wine_train, X_Wine_valid, y_Wine_train, y_Wine_valid = load_wine_cobaia(seed)
    Wisconsin_classes, X_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_train, y_Wisconsin_valid = load_Wisconsin_cobaia(seed)


    #montar arvores de decisão
    decisionTree_Wine = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Wine.fit(X_Wine_train, y_Wine_train)

    decisionTree_Wisconsin = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Wisconsin.fit(X_Wisconsin_train, y_Wisconsin_train)

    decisionTree_Iris = DecisionTreeClassifier(max_depth = 3, random_state = decisionTreeSeed)
    decisionTree_Iris.fit(X_Iris_train, y_Iris_train)

    #montar redes neurais

    #0 hidden layer
    Wine_NN_0h = NN.nnf([4, 3],[ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
    Wine_NN_0h.train(X_Wine_train, y_Wine_train, X_Wine_valid, y_Wine_valid, epochs=1000, learning_rate=0.01)

    Wine_0h_params = Wine_NN_0h.get_params()
    Wine_0h_U = Neurons_to_Lists(Wine_0h_params)

    Wine_0h_regras_1_KT = KT.KT_1(Wine_0h_U)
    Wine_0h_regras_2_MofN = MofN.MofN_2(Wine_0h_U, Wine_NN_0h, X_Wine_train, y_Wine_train)
    Wine_0h_regras_3_RuleExtractingLearning = REL.Rule_extraction_learning_3(Wine_NN_0h, Wine_classes, X_Wine_train)
    Wine_0h_regras_4_RxREN = RxREN.RxREN_4(Wine_NN_0h, Wine_0h_U,[],[], Wine_classes)

    Wisconsin_NN_0h = NN.nnf([30, 2],[ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
    Wisconsin_NN_0h.train(X_Wisconsin_train, y_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_valid, epochs=1000, learning_rate=0.01)

    Wisconsin_0h_params = Wisconsin_NN_0h.get_params()
    Wisconsin_0h_U = Neurons_to_Lists(Wisconsin_0h_params)

    Wisconsin_0h_regras_1_KT = KT.KT_1(Wisconsin_0h_U)
    Wisconsin_0h_regras_2_MofN = MofN.MofN_2(Wisconsin_0h_U, Wisconsin_NN_0h, X_Wisconsin_train, y_Wisconsin_train)
    Wisconsin_0h_regras_3_RuleExtractingLearning = REL.Rule_extraction_learning_3(Wisconsin_NN_0h, Wisconsin_classes, X_Wisconsin_train)
    Wisconsin_0h_regras_4_RxREN = RxREN.RxREN_4(Wisconsin_NN_0h, Wisconsin_0h_U,[],[], Wisconsin_classes)

    Iris_NN_0h = NN.nnf([13, 3],[ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
    Iris_NN_0h.train(X_Iris_train, y_Iris_train, X_Iris_valid, y_Iris_valid, epochs=1000, learning_rate=0.01)

    Iris_0h_params = Iris_NN_0h.get_params()
    Iris_0h_U = Neurons_to_Lists(Iris_0h_params)

    Iris_0h_regras_1_KT = KT.KT_1(Iris_0h_U)
    Iris_0h_regras_2_MofN = MofN.MofN_2(Iris_0h_U, Iris_NN_0h, X_Iris_train, y_Iris_train)
    Iris_0h_regras_3_RuleExtractingLearning = REL.Rule_extraction_learning_3(Iris_NN_0h, Iris_classes, X_Iris_train)
    Iris_0h_regras_4_RxREN = RxREN.RxREN_4(Iris_NN_0h, Iris_0h_U,[],[], Iris_classes)

    #1 hidden layer

    Wine_model_cases_n1 = load_models_params(X_Wine_train, y_Wine_train, X_Wine_valid, y_Wine_valid, 4, 3)
    Wisconsin_model_cases_n1 = load_models_params(X_Wisconsin_train, y_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_valid, 30, 2)
    Iris_model_cases_n1 = load_models_params(X_Iris_train, y_Iris_train, X_Iris_valid, y_Iris_valid, 13, 3)

    #2 hidden layers

    Wine_model_cases_n2 = load_models_params(X_Wine_train, y_Wine_train, X_Wine_valid, y_Wine_valid, 4, 3, nLayer = 2)
    Wisconsin_model_cases_n2 = load_models_params(X_Wisconsin_train, y_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_valid, 30, 2, nLayer = 2)
    Iris_model_cases_n2 = load_models_params(X_Iris_train, y_Iris_train, X_Iris_valid, y_Iris_valid, 13, 3, nLayer = 2)

    #3 hidden layers

    Wine_model_cases_n3 = load_models_params(X_Wine_train, y_Wine_train, X_Wine_valid, y_Wine_valid, 4, 3, nLayer = 3)
    Wisconsin_model_cases_n3 = load_models_params(X_Wisconsin_train, y_Wisconsin_train, X_Wisconsin_valid, y_Wisconsin_valid, 30, 2, nLayer = 3)
    Iris_model_cases_n3 = load_models_params(X_Iris_train, y_Iris_train, X_Iris_valid, y_Iris_valid, 13, 3, nLayer = 3)


    return

#algoritmo_1_KT()
algoritmo_2_MofN() #problema no tratamento de clusters
#algoritmo_3_RuleExtractLearning()
#algoritmo_4_RxRen()
#single_function_test()
print("bateria de teste terminado")