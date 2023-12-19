import Node
import NeuralNetwork
import numpy as np
import pandas as pd

def Subset(classNN, rules, examples):
#try to classify with the ruleset and compare with class label
    result = True
    for e in examples:
        ruleLabel = classify(rules, e)
        result = result and (ruleLabel == classNN)
    return result

def classify(R, E):
#classify the example with the rules and to a class, consider drop parameter C
#with each rule pass the examples and check the true and false for each class
    if R is None:
        return "no_input_value"

    prediction = R.step(E)

    return prediction

def covered(rule, example, c):
#check if the example is covered by the rule
#try to classify with the rule?
    result = classify(rule, example)

    return c == result

def possible_values(examplesData):
#valores possiveis para os exemplos
    result = dict()
    #print("dimensões: %s" % (examplesData.shape))

    feature_length = examplesData.shape[1]
    for i in range(feature_length):
        result[i] = []

    for e in examplesData:
        for i in range(feature_length):
            result[i].append(e[i])

    for idx, r in result.items():
        result[idx] = list(set(r))

    return result

def make_examples(possibilities, n = 1):
    result = []

    for i in range(n):
        oneSample = []

        for e in possibilities.values():
            indexRange = len(e)
            choice = np.random.randint(indexRange)
            oneSample.append(e[choice])

        result.append(oneSample)

    return result

def filter(antecendents, ant_to_null):

    hold = None
    hold_idx = -1

    for idx, ant in enumerate(antecendents):
        if ant.threshold == ant_to_null and ant.comparison == ant_to_null and ant.featureIndex == ant.featureIndex:
            hold = ant
            hold_idx = idx

    if hold_idx != -1:
        copy_tree = antecendents[-1].copy()
        copy = copy_tree.getAntecedent()
        hold = copy[hold_idx]
        left_branch = hold.left
        right_branch = hold.right

        side = hold[0]
        origin = hold[1]

        if side == 0:
            numSons = hold.num_sons()
            if numSons == 2:
                resultNode = hold.rotation45()
                return resultNode

            elif numSons == 1:
                if left_branch:
                    result = hold.left
                else:
                    result = hold.right
                return result

            else:
                return None

            copy.pop(hold_idx)
            return copy_tree

        if not left_branch and not right_branch:
            #delete hold from list and tree
            if side == -1:
                origin.set_left(None)

            elif side == 1:
                origin.set_right(None)

            copy.pop(hold_idx)
            return copy_tree

        elif not left_branch:
            if side == -1:
                origin.set_left(hold.right)

            elif side == 1:
                origin.set_right(hold.right)

            copy.pop(hold_idx)
            return copy_tree

        elif not right_branch:
            if side == -1:
                origin.set_left(hold.left)

            elif side == 1:
                origin.set_right(hold.left)

            copy.pop(hold_idx)
            return copy_tree

        else:
            resultNode = hold.rotation45()
            if side == -1:
                origin.set_left(resultNode)

            elif side == 1:
                origin.set_right(resultNode)

            copy.pop(hold_idx)
            return copy_tree

    return antecendents[-1][2]

def conjuntive_rule(Exemplo, endResult):
    resultRule = Node.Node(featureIndex=0, threshold=Exemplo[0])
    previousPremisse = resultRule
    for idx in range(1, len(Exemplo)):
        partial_premisse = Node.Node(featureIndex=idx, threshold=Exemplo[idx])
        previousPremisse.set_right(partial_premisse)
        previousPremisse = partial_premisse

    previousPremisse.set_right(endResult)

    return resultRule

def label_code_block(R, E, C):

    c = classify(R, E)

    if not covered(R, E, C):
        r = conjuntive_rule(E, R[C])
        if r:
            print("conjuntive rule made for %s:"%(C))
            #r.print()
        else:
            print("conjuntive rule not made")

        ant_r = r.getAntecedent()
        print("number of antecendents: %d" % (len(ant_r)))
        if ant_r:
            for ri in ant_r:
                r_ = filter(ant_r, ri)
                if r:
                    print("filtered rule made for %s:"%(C))
                    #r.print()
                else:
                    print("filtered rule not made")

                if Subset(c,r_):
                    r = r_

            if R is None:
                R = r

            else:
                r.set_left(R)
                R = r
                if R:
                    print("updated rule made for %s:"%(C))
                    #r.print()
                else:
                    print("updated rule not made")

    return R

def Rule_extraction_learning_3(M, C, Ex, theta = 0):
    R = dict() #organizado por c lista de nós com ramos conectados e listados
    for c in C:
        #TODO: trocar a folha por uma regra feita por um exemplo de uma classe
        R[c] = None

    Possibilities = possible_values(Ex)
    numClasses = len(C)
    outputLayerIndex = M.get_params()["num layers"] - 1

    voltas = 0

    print("Iniciou regras e possibilidades")
    print("numero de labels: %d" % (numClasses))
    for idx, c in R.items():
        print("label: {}".format(idx))
        print(c)

    while voltas < numClasses:
        print("numero de voltas: %d" % (voltas))
        voltas += 1
        qtd_exemplos = numClasses * voltas
        E = make_examples(Possibilities, n = qtd_exemplos*1000)

        O = []
        Sum_IO = []
        for example in E:
            model_result = M.predict(np.squeeze(example))
            Sum_IO.append(sum(M.get_params()["Z"+str(outputLayerIndex)]))
            O.append(C[np.argmax(model_result)])

        print("exemplos gerados: %d" % (len(E)))

        for idx, s in enumerate(Sum_IO):
            for neuron in s:
                ModelOutput =  O[idx]
                if neuron > theta:
                    #Todo: change function call, consider saving versions of examples devided by the outputs

                    R[ModelOutput] = label_code_block(R[ModelOutput], E[idx], ModelOutput)

            else:
                for i in range(len(E[idx])):
                    for v in Possibilities[i]:
                        temp = E[idx]
                        temp[i] = v
                        #changing the value of ei to vij increase s
                        modelResult = M.predict(np.array(temp))
                        newSum = sum(M.get_params()["Z"+str(outputLayerIndex)])

                        if  newSum > s:
                            E[idx][i] = v
                            O[idx] = C[np.argmax(modelResult)]
                            Sum_IO[idx] = newSum

                        if s > theta:
                            R[O[idx]] = label_code_block(R[O[idx]], E[idx], O[idx])

    return R