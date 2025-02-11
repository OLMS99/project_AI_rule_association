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
    print(predictions)
    print(C)
    return MM.Compute_Acc_naive(predictions, C)

def getSetSize(qSet):
    total = 0
    for g in qSet:
        if len(g) <= 0:
            continue
        total+=1

    return total

def createGroup(wrongInstances, targetClass, debug = False):
    if debug:
        print("#########################################")
        print("targetClass: %s" % (targetClass))
        print("-----------------------------------------")
        print("instances: %s" % (len(wrongInstances)))
    group = []
    expectedClass = targetClass
    classesInInstances = set()
    for e in wrongInstances:

        predictionCase = e[1]
        classesInInstances.add(predictionCase)
        if predictionCase != expectedClass:
            continue
        group.append(e)
    if debug:
        print("instances in group: %s" % (len(group)))
        print("*****************************************")
        print("classes in the instances before grouping: %s" % (classesInInstances))
        print("searched class: %s" % (expectedClass))
        print("#########################################")
    return group

def makerule_RxREN(minVal, maxVal, neuron_idx):
    if maxVal == float('-inf') and minVal == float('inf'):
        return

    print("valor máximo: %s valor mínimo: %s" % (maxVal, minVal))
    print("neurônio de entrada: %s" % (neuron_idx))
    node1 = Node.Node(featureIndex = neuron_idx, threshold = maxVal, comparison = "<=")
    node2 = Node.Node(featureIndex = neuron_idx, threshold = minVal, comparison = ">=")
    node1.set_right(node2)

    return node1

def check_prediction(pred, y):
    return np.argmax(np.round(pred)) == np.argmax(y)

def formSet(groups, erri, alpha, debug = False):
    if debug:
        print("#########################################")
        print("erri: %s" % (erri))
        print("-----------------------------------------")
        print("tamanho grupos: %s" % ([len(g) for g in groups]))

    Q=[]
    for i, g in enumerate(groups):
        size = len(g)
        if debug:
            print("%s * %s > %s * %s" %(size, size, alpha, erri))
        if size * size > alpha * erri:
            Q.append(g)
        else:
            Q.append([])

    if debug:
        print("tamanho sets de Q: %s" % ([len(q) for q in Q]))

    return Q

def getInputValueArrayQ(Q, inputIdx, classIdx):
    result = []
    for G in Q:
        for g in G:
            if len(g) == 0:
                continue
            if g[1] != classIdx:
                continue
            result.append(g[0][inputIdx])
    return result

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

                item = (case, np.argmax(y[number]), np.argmax(prediction))
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
            print("Pacc < Nacc - 0.01")
            print("%s < %s - 0.01" % (Pacc, Nacc))

        if Pacc < (Nacc - 0.01):
            local_NN = NN_
            mapL = L_
            input_size = len(L_)
            #go to top code block
        else:
            break

    #montando matrizes
    print("iniciando fase de montagem")
    print("organizando valores")
    m = len(mapL)
    n = len(C)
    g = [[[] for k in range(n)] for j in range(m)]
    leng = [[0 for k in range(n)] for j in range(m)]
    q = [[] for j in range(m)]
    lenq = [0 for j in range(m)]
    minMatrix = np.full((m,n), float('inf'))
    maxMatrix = np.full((m,n), float('-inf'))

    for i, l in enumerate(mapL):
        for k, c in enumerate(C):
            print("group for input: %s class: %s" % (l, c))
            g[i][k] = createGroup(E[l], c)
            leng[i][k] = len(g[i][k])

        #alpha value [0.1,0.5]
        q[i] = formSet([g[i][k] for k in range(len(C))], err[l], alpha)
        lenq[i] = getSetSize(q[i])
        for k, c in enumerate(C):
            print("group input: %s class: %s" % (l, c))
            print("set size: %s" % (lenq[i]))
            if lenq[i] <= 0:
                continue

            possibleValues = getInputValueArrayQ(q[i], i, k)
            if len(possibleValues) <= 0:
                continue

            minMatrix[i][k] = min(min(possibleValues), minMatrix[i][k])
            maxMatrix[i][k] = max(max(possibleValues), maxMatrix[i][k])

    print("grupos")
    print(leng)
    print("tamanhos dos conjuntos Q")
    print(lenq)
    print("valores mínimos")
    print(minMatrix)
    print("valores máximos")
    print(maxMatrix)
    #extraindo regras
    print("extraindo regras")
    print("numero de neurônios de entrada conectados")
    print(len(mapL))
    for k, c in enumerate(C):
        cn = None
        for i, l in enumerate(mapL):
            cnj = None
            if len(g[i][k]) > alpha * err[l]:
                #create node based on this expression
                #cnj = (mapL[i] >= minMatrix[i][k]) and (mapL[i] <= maxMatrix[i][k])
                if debug:
                    print("montando regra do neurônio: %s, classe %s" % (l, c))
                    print("valor Máximo: %s Valor Mínimo: %s" % (maxMatrix[i][k],minMatrix[i][k]))
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
        print("==================================")
    print(classRuleSets)

def isComplete(RxRENruleSet):
    if RxRENruleSet is None:
        return False
    for classLabel, classRules in RxRENruleSet.items():
        if classRules is []:
            return False
    return True

def delete(RxRENruleSet):
    if RxRENruleSet is None:
        return
    RxRENruleSet.clear()