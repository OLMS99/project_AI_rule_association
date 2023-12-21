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
        return "no_rule_here"

    prediction = R.step(E)

    return prediction

def covered(rule, example, c, debug=False):
#check if the example is covered by the rule
#try to classify with the rule?
    result = classify(rule, example)

    if debug:
        print("{0} == {1}".format(c,result))

    if result == "no_rule_here":
        return False
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

def conjuntive_rule(Exemplo, endResult, leaf, debug = False):
    presence_array = [False] * len(Exemplo)
    current_node = endResult
    hook = None

    if endResult:
        while not current_node.is_leaf_node():
            judge = current_node.eval(Exemplo)
            if judge:
                presence_array[current_node.featureIndex] = True
                current_node = current_node.right

            else:
                if current_node.left:
                    current_node = current_node.left
                else:
                    hook = current_node
                    break

    resultRule = None
    previousPremisse = None

    for idx, feature_check in enumerate(presence_array):
        if not feature_check:
            partial_premisse = Node.Node(featureIndex=idx, threshold=Exemplo)

            if not resultRule:
                resultRule = partial_premisse
            else:
                previousPremisse.set_right(partial_premisse)

            previousPremisse = partial_premisse

    previousPremisse.set_right(leaf)

    if hook:
        hook.set_left(resultRule)

    if debug:
        resultRule.print()

    if endResult:
        return endResult
    return resultRule

def label_code_block(R, E, debug = False):

    c = classify(R, E)
    is_covered = covered(R, E, c)
    if debug:
        print(c)
        if is_covered:
            print("regra atual cobre exemplo")
        else:
            print("regra atual NÃO cobre exemplo")

    if not is_covered:

        leaf = Node.Node(value=c)
        r = conjuntive_rule(E, R, leaf)

        if debug:
            if r:
                print("conjuntive rule made for %s:"%(c))
                r.print()
            else:
                print("conjuntive rule not made")

        ant_r = r.getAntecedent()

        if debug:
            print("number of antecendents: %d" % (len(ant_r)))

        if ant_r:
            for ri in ant_r:
                r_ = filter(ant_r, ri)
                if debug:
                    if r:
                        print("filtered rule made for %s:"%(c))
                        r.print()
                    else:
                        print("filtered rule not made")

                if Subset(c,r_):
                    r = r_

            if R is None:
                R = r

            else:
                r.set_left(R)
                R = r
                if debug:
                    if R:
                        print("updated rule made for %s:"%(C))
                        r.print()
                    else:
                        print("updated rule not made")

    return R

def Rule_extraction_learning_3(M, C, Ex, theta = 0, debug = False):
    R = dict() 
    for c in C:
        #TODO: trocar a folha por uma regra feita por um exemplo de uma classe
        R[c] = None

    Possibilities = possible_values(Ex)
    numClasses = len(C)

    modelParams = M.get_params()
    outputLayerIndex = modelParams["num layers"] - 1
    weightOutputLayer = modelParams["W"+str(outputLayerIndex)]
    biasOutputLayer = modelParams["b"+str(outputLayerIndex)]

    voltas = 0

    if debug:
        print("Iniciou regras e possibilidades")
        print("numero de labels: %d" % (numClasses))
        for idx, c in R.items():
            print("label: {}".format(idx))
            print(c)

    while voltas < numClasses:
        if debug:
            print("numero de voltas: %d" % (voltas))
        voltas += 1
        qtd_exemplos = numClasses * voltas
        E = make_examples(Possibilities, n = qtd_exemplos*1000)

        O = []
        Sum_IO = []
        for example in E:
            model_result = M.predict(np.squeeze(example))
            inputToOutput = M.get_params()["A"+str(outputLayerIndex-1)]
            if debug:
                print(inputToOutput)
            Sum_IO.append(sum(inputToOutput))
            O.append(C[np.argmax(model_result)])

        if debug:
            print("exemplos gerados: %d" % (len(E)))

        for idx, s in enumerate(Sum_IO):
            for neuron in s:
                ModelOutput =  O[idx]
                if debug:
                    print("{0} > {1}".format(neuron, theta))
                if neuron > theta:
                    #Todo: change function call, consider saving versions of examples devided by the outputs

                    R[ModelOutput] = label_code_block(R[ModelOutput], E[idx])

                else:
                    for i in range(len(E[idx])):
                        for v in Possibilities[i]:
                            temp = E[idx]
                            temp[i] = v
                            #changing the value of ei to vij increase s?
                            modelResult = M.predict(np.array(temp))
                            newSum = sum(M.get_params()["A"+str(outputLayerIndex-1)])

                            if  newSum > s:
                                E[idx][i] = v
                                O[idx] = C[np.argmax(modelResult)]
                                Sum_IO[idx] = newSum

                            if s > theta:
                                R[O[idx]] = label_code_block(R[O[idx]], E[idx])

    return R