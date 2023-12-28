import Node
import NeuralNetwork as ANN
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
#classify the example with the rules and to a class
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
    #from a list of antecendents, filter the desired antecendent
    #return the modified copy of the tree
    hold_idx = -1
    copyTree = antecendents[-2][1].copy()

    for idx, ant in enumerate(antecendents):
        premisse = ant[2]
        if premisse == ant_to_null:
            hold_idx = idx
            side = ant[0]
            break

    if hold_idx == -1:
        #antecendent not found
        return copyTree

    copy_ant = copyTree.getAntecendent()
    origin = copy_ant[hold_idx][1]

    if side == 0:
        target = copy_ant[-2][1]

    elif side == -1:
        target = origin.left

    elif side == 1:
        target = origin.right

    new_branch = target.rotation45()

    #connectar origem da variavel target com o ramo resultante da rotação 45

    if side == 0:
        return new_branch

    elif side == -1:
        origin.set_left(new_branch)

    elif side == 1:
        origin.set_right(new_branch)

    return copyTree

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
            partial_premisse = Node.Node(featureIndex=idx, threshold=Exemplo[idx])

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

def label_code_block(R, E, true_result, debug = False):

    is_covered = covered(R, E, true_result, debug=debug)
    if debug:
        print(true_result)
        if is_covered:
            print("regra atual cobre exemplo")
        else:
            print("regra atual NÃO cobre exemplo")

    if not is_covered:

        leaf = Node.Node(value=true_result)
        r = conjuntive_rule(E, R, leaf, debug=debug)

        if debug:
            if r:
                print("conjuntive rule made for %s:"%(true_result))
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
                        print("filtered rule made for %s:"%(true_result))
                        r.print()
                    else:
                        print("rule filtered entirely")

                if Subset(true_result,r_,E):
                    r = r_

            if R is None:
                R = r

            else:
                r.set_left(R)
                R = r
                if debug:
                    if R:
                        print("updated rule made for %s:"%(true_result))
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
            inputToOutput = M.get_params()["Z"+str(outputLayerIndex)]

            Sum_IO.append(inputToOutput[np.argmax(model_result)])
            output_ONeuron = np.argmax(model_result)
            O.append([output_ONeuron, C[output_ONeuron]])

        if debug:
            print("exemplos gerados: %d" % (len(E)))

        for idx, s in enumerate(Sum_IO):
            for neuron in s:
                ModelOutput =  O[idx][1]
                #if debug:
                #    print("{0} > {1}".format(neuron, theta))
                if neuron > theta:
                    #Todo: change function call, consider saving versions of examples devided by the outputs

                    R[ModelOutput] = label_code_block(R[ModelOutput], E[idx], ModelOutput, debug=debug)

                else:
                    for i in range(len(E[idx])):
                        for v in Possibilities[i]:
                            temp = E[idx]
                            temp[i] = v
                            #changing the value of ei to vij increase s?
                            modelResult = M.predict(np.array(temp))
                            newSum = M.get_params()["Z"+str(outputLayerIndex)][O[idx][0]]

                            if  newSum > s:
                                E[idx][i] = v
                                Sum_IO[idx] = newSum

                            if s > theta:
                                R[O[idx][1]] = label_code_block(R[O[idx][1]], E[idx],O[idx][1])

    return R