import numpy as np

import Utils
import Node

import NeuralNetwork as NN
import ModelMetrics as MM

def computeAccuracy(network, dataset, C):
    predictions  = []
    for t in dataset:
        predictions.append(network.predict(t))
    return MM.Compute_Acc_naive(predictions, C)

def createGroup(wrongInstances, targetClass):
    group = []
    expectedAnswer = np.argmax(targetClass)
    for e in wrongInstances:
        predictionAnswer = np.argmax(e[1])
        if predictionAnswer == expectedAnswer:
            group.append(e)
    return group

def makerule_RxREN(minVal, maxVal, neuron_idx):
    node1 = Node.Node(featureIndex = neuron_idx, threshold = maxVal, comparison = "<=")
    node2 = Node.Node(featureIndex = neuron_idx, threshold = minVal, comparison = ">=")
    node1.set_right(node2)

    return node1

def check_prediction(pred, y):
    return np.argmax(np.round(pred)) == np.argmax(y)

def formSet(groups, erri, alpha):
    Q=[]
    for i, g in enumerate(groups):
        size = len(g)
        if size * size > alpha * erri:
            Q.append(g)
    return Q

def lenElem(setQi):
    result = []
    for qi in setQi:
        result.append(len(qi))

    return result

#ideia: L, H, N -> H, L = H[0], N = H[-1] H = H/H[0] and H[-1]
#T = exemplos que a rede neural classificou corretamente
#y = resultados esperados
#alpha = variavel do algoritmo, periodo de valor [0.1, 0.5]
def RxREN_4(M, H, T, y, C, alpha = 0.1, debug = False):
    local_NN = M

    input_size = len(H[0][0][0])
    mapL = []
    for i in range(input_size):
        mapL.append(i)

    R = dict()
    for c in C:
        R[c] = []

    B = []
    E = dict()
    err = dict()

    #Top block of code
    while True:
        B = []
        E = dict()
        err = dict()

        for idx, l in enumerate(mapL):
            temp_network = local_NN.copy().prune_input([l])
            E[l] = []
            #test the classification
            for number, case in enumerate(T):
                prediction = temp_network.predict(case)
                if not check_prediction(prediction, y[number]):
                    item = (l, y[number], prediction, case)
                    E[l].append(item)

            #set of incorrectly classified instances of ANN without li on set of correctly classified instances

            err[l] = len(E[l])

            if debug:
                print("neuronio de entrada %s: numero de erros: %s" % (l, err[l]))

        theta = min(err.values())
        insig = MM.Where_n(err, n=theta)
        for li in insig:
            B.append(mapL[li])

        NN_ = local_NN.copy().prune_input(B)
        L_ = list(filter(lambda i: i not in B, mapL))
        Pacc = computeAccuracy(NN_, T, y)
        Nacc = computeAccuracy(local_NN, T, y)

        if debug:
            print("%s < %s - 0.01" % (Pacc, Nacc))

        if 100 * Pacc < (100 * Nacc - 1):
            local_NN = NN_
            mapL = L_
            input_size = len(list(mapL))
            #go to top code block
        else:
            break

    #montando matrizes

    m = len(mapL)
    n = len(C)
    g = [[[] for k in range(n)] for j in range(m)]
    minMatrix = [[float('inf') for k in range(n)] for j in range(m)]
    maxMatrix = [[float('-inf') for k in range(n)] for j in range(m)]

    for i, l in enumerate(mapL):
        for k, c in enumerate(C):
            g[i][k] = createGroup(E[l], c)
            #alpha value [0.1,0.5]
            Qi = formSet(g[i][k], err[l], alpha)
            lenQi = lenElem(Qi)
            minMatrix[i][k] = min(lenQi) if len(Qi) > 0 else float('inf')
            maxMatrix[i][k] = max(lenQi) if len(Qi) > 0 else float('-inf')

    #extraindo regras

    for k, c in enumerate(C):
        cn = None
        for i, l in enumerate(mapL):
            for idx, c in enumerate(C):
                cnj = None
                if len(g[i][k]) > alpha * err[l]:
                    #create node based on this expression
                    #cnj = (mapL[i] >= minMatrix[i][k]) and (mapL[i] <= maxMatrix[i][k])
                    cnj = makerule_RxREN(minMatrix[i][k], maxMatrix[i][k], l)

                if cn is None:
                    cn = cnj
                else:
                    #and -> set_right()
                    cn.append_right(cnj)

        if cn is not None:
            ck = Node.Node(value = c)
            cn.append_right(ck)
            R[c].append(cn)

    return R

def parseRules(classRuleSets, inputValues):
    resultBatch = []
    for ruleSet in classRuleSets.values():
        for rule in ruleSet:
            resultBatch.append(rule.step(inputValues))

        resultBatch = set(resultBatch)
        resultBatch = resultBatch.remove("no_output_values") if "no_output_values" in resultBatch else resultBatch
        resultBatch = list(resultBatch)

    return resultBatch if len(resultBatch) > 0 else ["no_results"]

def isComplete(RxRENruleSet):
    for classLabel, classRules in RxRENruleSet.items():
        if classRules is []:
            return False
    return True