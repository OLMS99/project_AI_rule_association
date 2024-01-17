import random
import math
import numpy as np
import time

import Utils
import Node
import NeuralNetwork as NN

def computeAccuracy(network, dataset, C):
    predictions  = []
    for t in dataset:
        predictions.append(network.predict(t))
    return Utils.Compute_Acc_naive(predictions, C)



def createGroup(wrongInstances, targetClass):
    group = []
    for e in wrongInstances:
        if e[1] == targetClass:
            group.append(e)
    return group

def formSet(groups, error, alpha):
    Q=[]
    for i, g in enumerate(groups):
        size = len(g)
        if size * size > alpha * error[i]:
            Q.append(g)
    return Q

#ideia: L, H, N -> H, L = H[0], N = H[-1] H = H/H[0] and H[-1]
#T = exemplos que a rede neural classificou corretamente
#y = resultado esperado
#alpha = variavel do algoritmo, periodo de valor [0.1, 0.5]
def RxREN_4(NN, H, T, y, alpha = 0.1):
    L = H[0]
    N = H[-1]
    B=[]
    R=[]
    C = y.unique()
    input_size = len(L)
    mapL = []
    for i in range(input_size):
        mapL.append(i)
    #Top block of code
    while True:
        B=[]
        E = dict()
        err = dict()
        for l in mapL:
            temp_network = NN.prune([l])
            #test the classification
            for number, case in enumerate(T):
                prediction = temp_network.predict(case)
                if prediction != y[number]:
                    item = (l, y[number], prediction)
                    if l in E:
                        E[l].append(item)
                    else:
                        E[l] = [item]
                    #set of incorrectly classified instances of ANN without li on set of correctly classified instances
            err[l] = len(E[l])

        m = input_size
        theta = err.min()
        insig = Utils.Where_n(err, n=theta)
        for li in insig: 
            B.append(li)

        NN_ = NN.prune(B)
        L_ = filter(lambda i: i not in B, mapL)
        Pacc = computeAccuracy(NN_)
        Nacc = computeAccuracy(NN)
        if Pacc >= (Nacc - 1):
            NN = NN_
            mapL = L_
            input_size = len(mapL)
            #go to top code block
        else:
            break

    m = len(L)
    Q = [[]] * m
    g = [[[] for k in range(m)] for j in range(len(C))]
    minMatrix = [[[] for k in range(m)] for j in range(len(C))]
    maxMatrix = [[[] for k in range(m)] for j in range(len(C))]
    for i, l in enumerate(L):
        for k, c in enumerate(C):
            g[i][k] = createGroup(E[l], c)

            #alpha value [0.1,0.5]
            Q[i].append(formSet(g[i][k], err, alpha))
            minMatrix[i][k] = min(Q[i])
            maxMatrix[i][k] = max(Q[i])

    for k in range(n):
        j = 1
        for i in m: 
            for k, c in enumerate(C):
                if len(g[i][k]) > alpha * err[i]:
                    #create node based on this expression
                    cnj = (data(L[i]) >= minMatrix[i][k]) and (data(L[i]) <= maxMatrix[i][k])

                if j==1:
                    cn = cnj
                else:
                    #and -> set_right()
                    cn.append_right(cnj)

                j+=1
        #or -> set_left()
        if cn:
            cn.append_right(ck)
            R.append_left(cn)

    return R