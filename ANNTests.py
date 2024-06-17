from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import random
import math
import numpy as np
import time

import ActivationFunctions as ACT
import LossFunctions as Loss
import ModelMetrics as mmetrics
import NeuralNetwork as NN

def teste_OR(seed):
    #testando rede no caso OR 

    casos_OR = np.array([[1,1],[1,0],[0,1],[0,0]])
    respostas_OR = np.array([[1],[1],[1],[0]])

    OR_dnn = NN.nnf([2, 2, 1],[ACT.ReLU, ACT.ReLU, ACT.sigmoid], Loss.naive_loss, Loss.naive_loss_prime, seed = seed)
    OR_dnn.train(casos_OR, respostas_OR, casos_OR, respostas_OR,epochs=100, learning_rate = 0.01)
    OR_dnn_params = OR_dnn.get_params()

    pred_OR_0 = OR_dnn.predict(casos_OR[0])
    pred_OR_1 = OR_dnn.predict(casos_OR[1])
    pred_OR_2 = OR_dnn.predict(casos_OR[2])
    pred_OR_3 = OR_dnn.predict(casos_OR[3])

    print("caso 1 OR 1 machine: %s (%s) answer: %s" % (int(pred_OR_0 > 0.5), pred_OR_0, respostas_OR[0]))
    print("caso 1 OR 0 machine: %s (%s) answer: %s" % (int(pred_OR_1 > 0.5), pred_OR_1, respostas_OR[1]))
    print("caso 0 OR 1 machine: %s (%s) answer: %s" % (int(pred_OR_2 > 0.5), pred_OR_2, respostas_OR[2]))
    print("caso 0 OR 0 machine: %s (%s) answer: %s" % (int(pred_OR_3 > 0.5), pred_OR_3, respostas_OR[3]))

def teste_XOR(seed):
    #testando rede no caso XOR 
    casos_XOR = np.array([[1,1],[1,0],[0,1],[0,0]])
    respostas_XOR = np.array([[0],[1],[1],[0]])

    XOR_dnn = NN.nnf([2, 2, 1],[ACT.ReLU, ACT.ReLU, ACT.sigmoid], Loss.naive_loss, Loss.naive_loss_prime, seed = seed)
    XOR_dnn.train(casos_XOR, respostas_XOR, casos_XOR, respostas_XOR, epochs=1000, learning_rate=0.01)
    XOR_dnn_params = XOR_dnn.get_params()

    pred_XOR_0 = XOR_dnn.predict(casos_XOR[0])
    pred_XOR_1 = XOR_dnn.predict(casos_XOR[1])
    pred_XOR_2 = XOR_dnn.predict(casos_XOR[2])
    pred_XOR_3 = XOR_dnn.predict(casos_XOR[3])

    print("caso 1 XOR 1 machine: %s (%s) answer: %s" % (int(pred_XOR_0 > 0.5), pred_XOR_0, respostas_XOR[0]))
    print("caso 1 XOR 0 machine: %s (%s) answer: %s" % (int(pred_XOR_1 > 0.5), pred_XOR_1, respostas_XOR[1]))
    print("caso 0 XOR 1 machine: %s (%s) answer: %s" % (int(pred_XOR_2 > 0.5), pred_XOR_2, respostas_XOR[2]))
    print("caso 0 XOR 0 machine: %s (%s) answer: %s" % (int(pred_XOR_3 > 0.5), pred_XOR_3, respostas_XOR[3]))


def teste_iris(seed):
    dataset = load_iris()
    data = dataset.data

    lb = preprocessing.LabelBinarizer()
    target = lb.fit_transform(dataset.target)

    split_train_size = 0.7
    split_test_size = 1 - split_train_size

    train_X,valid_X,train_y,valid_y = train_test_split(data, target, test_size=split_test_size, random_state=13)

    Iris_dnn = NN.nnf([4, 5, 3],[ACT.sigmoid, ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
    Iris_dnn.train(train_X,train_y,valid_X,valid_y, epochs=1000, learning_rate=0.01)

    y_pred_train = np.zeros(shape=(train_X.shape[0], Iris_dnn.output_size))
    for i, sample in enumerate(train_X):
        y_pred_train[i] = Iris_dnn.predict(sample)

    y_pred_valid = np.zeros(shape=(valid_X.shape[0], Iris_dnn.output_size))
    for i, sample in enumerate(valid_X):
        y_pred_valid[i] = Iris_dnn.predict(sample)

def teste_Winsconsin(seed):
    dataset = load_breast_cancer()
    data = dataset.data

    lb = preprocessing.LabelBinarizer()
    target = lb.fit_transform(dataset.target)

    split_train_size = 0.7
    split_test_size = 1 - split_train_size

    train_X,valid_X,train_y,valid_y = train_test_split(data, target, test_size=split_test_size, random_state=13)

    Wisconsin_dnn = NN.nnf([30, 31, 2],[ACT.sigmoid, ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
    Wisconsin_dnn.train(train_X,train_y,valid_X,valid_y, epochs=1000, learning_rate=0.01)

    y_pred_train = np.zeros(shape=(train_X.shape[0], Wisconsin_dnn.output_size))
    for i, sample in enumerate(train_X):
        y_pred_train[i] = Wisconsin_dnn.predict(sample)

    y_pred_valid = np.zeros(shape=(valid_X.shape[0], Wisconsin_dnn.output_size))
    for i, sample in enumerate(valid_X):
        y_pred_valid[i] = Wisconsin_dnn.predict(sample)

def teste_wine(seed):
    dataset = load_wine()
    data = dataset.data

    lb = preprocessing.LabelBinarizer()
    target = lb.fit_transform(dataset.target)

    split_train_size = 0.7
    split_test_size = 1 - split_train_size

    train_X,valid_X,train_y,valid_y = train_test_split(data, target, test_size=split_test_size, random_state=13)

    Wine_dnn = NN.nnf([13, 14, 3],[ACT.sigmoid, ACT.sigmoid, ACT.sigmoid], Loss.binary_cross_entropy, Loss.binary_cross_entropy_prime, seed = seed)
    Wine_dnn.train(train_X,train_y,valid_X,valid_y, epochs=1000, learning_rate=0.01)

    y_pred_train = np.zeros(shape=(train_X.shape[0], Wine_dnn.output_size))
    for i, sample in enumerate(train_X):
        y_pred_train[i] = Wine_dnn.predict(sample)

    y_pred_valid = np.zeros(shape=(valid_X.shape[0], Wine_dnn.output_size))
    for i, sample in enumerate(valid_X):
        y_pred_valid[i] = Wine_dnn.predict(sample)