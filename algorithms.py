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

#ideia: L, H, N -> H, H[0] = L, H[-1] = N
def RxREN_4(NN, L, H, N, T, y, alpha = 0.1):
    B=[]
    R=[]
    C = y.unique()
    #Top block of code
    while True:
        B=[]
        E = [[]] * len(L)
        err = [0] * len(L)
        for i, l in enumerate(L):
            temp_network = NN.prune([l])
            #test the classification
            for number, case in enumerate(T):
                prediction = temp_network.predict(case)
                if prediction != y[number]:
                    E[i].append((l, y[number], prediction))
                    #set of incorrectly classified instances of ANN without li on set of correctly classified instances
            err[i] = len(E[i])

        m = len(L)
        theta = err.min()
        insig = Utils.Where_n(err, n=theta)
        for li in insig: 
            B.append(li)

        NN_ = NN.prune(B)
        L_ = filter(lambda i: i not in B, L)
        Pacc = computeAccuracy(NN_)
        Nacc = computeAccuracy(NN)
        if Pacc >= (Nacc - 1):
            NN = NN_
            L = L_
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
            g[i][k] = createGroup(E[i], c)

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