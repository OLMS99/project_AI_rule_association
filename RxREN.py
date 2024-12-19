import numpy as np

import Utils
import Node

import NeuralNetwork as NN
import ModelMetrics as MM
import gc

def computeAccuracy(network, dataset, C):
    predictions  = []
    for t in dataset:
        predictions.append(network.predict(t))
    return MM.Compute_Acc_naive(predictions, C)

def createGroup(wrongInstances, targetClass):
    print("#########################################")
    print("targetClass: %s" % (targetClass))
    print("-----------------------------------------")

    group = []
    expectedAnswer = np.argmax(targetClass)
    for e in wrongInstances:

        predictionAnswer = int(np.argmax(np.round(e[2])))
        predictionCase = int(np.argmax(e[1]))
        print("prediction: %s   case: %s" % (predictionAnswer, predictionCase))

        if predictionAnswer != expectedAnswer:
            continue
        if predictionCase == expectedAnswer:
            group.append(e)

    print("group: %s" % (group))
    print("#########################################")
    return group

def makerule_RxREN(minVal, maxVal, neuron_idx):
    if maxVal == float('-inf') and minVal != float('inf'):
        return
#    if maxVal == float('-inf'):
#        node2 = Node.Node(featureIndex = neuron_idx, threshold = minVal, comparison = ">=")
#        return node2
#    if minVal != float('inf'):
#        node1 = Node.Node(featureIndex = neuron_idx, threshold = maxVal, comparison = "<=")
#        return node1

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
    mapL = [i for i in range(input_size)]

    R = dict()
    for c in C:
        R[c] = []

    B = []
    E = dict()
    err = dict()

    #Top block of code
    print("iniciando fase de poda")
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
                if check_prediction(prediction, y[number]):
                    continue

                item = (case, y[number], prediction)
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
            print("Pacc < Nacc - alpha")
            print("%s < %s - %s" % (Pacc, Nacc, alpha/10))

        if 100 * Pacc < (100 * Nacc - 1):
            local_NN = NN_
            mapL = L_
            input_size = len(mapL)
            #go to top code block
        else:
            break

    #montando matrizes
    print("iniciando fase de montagem")
    print("organizando valores")
    m = len(mapL)
    n = len(C)
    g = [[[] for k in range(n)] for j in range(m)]
    q = [[[] for k in range(n)] for j in range(m)]
    lenq = [[ 0 for k in range(n)] for j in range(m)]
    minMatrix = np.full((m,n), float('inf'))
    maxMatrix = np.full((m,n), float('-inf'))

    for i, l in enumerate(mapL):
        for k, c in enumerate(C):
            g[i][k] = createGroup(E[l], c)
            #alpha value [0.1,0.5]
            q[i][k] = formSet(g[i][k], err[l], alpha)
            lenq[i][k] = len(q[i][k])
            print("group input: %s class: %s" % (l,c))
            #print("group: %s" % (g[i][k]))
            #print("set: %s" % (Qi))
            #print("set size: %s" % (lenQi))
            if lenq[i][k] > 0:
                minMatrix[i][k] = min(min(Qi), minMatrix[i][k])
                maxMatrix[i][k] = max(max(Qi), maxMatrix[i][k])
    print("grupos")
    print(g)
    print("conjuntos Q")
    print(q)
    print("tamanhosdos conjuntos Q")
    print(lenq)
    print("valores mínimos")
    print(minMatrix)
    print("valores máximos")
    print(maxMatrix)
    #extraindo regras
    print("extraindo regras")
    for k, c in enumerate(C):
        cn = None
        for i, l in enumerate(mapL):
            for idx, cm in enumerate(C):
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
            if rule is None:
                continue
            resultBatch.append(rule.step(inputValues))

        resultBatch = set(resultBatch)
        resultBatch = resultBatch.remove("no_output_values") if "no_output_values" in resultBatch else resultBatch
        resultBatch = list(resultBatch)

    return resultBatch if len(resultBatch) > 0 else ["no_results"]

def printRules(classRuleSets):
    for c, r in classRuleSets.items():
        print("============== %s ================" % (c))
        if len(r) <= 0:
            print("nenhuma regra feita")
        else:
            for rule in r:
                rule.print()
    print(classRuleSets)

def isComplete(RxRENruleSet):
    for classLabel, classRules in RxRENruleSet.items():
        if classRules is []:
            return False
    return True

def delete(RxRENruleSet):
    RxRENruleSet.clear()