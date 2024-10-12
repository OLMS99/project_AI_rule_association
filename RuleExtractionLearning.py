import Node
import NeuralNetwork as ANN
import numpy as np
import pandas as pd
import gc

def Subset(classNN, rules, example):
#try to classify with the ruleset and compare with class label
    ruleLabel = classify(rules, example)
    result = (ruleLabel == classNN)
    return result

def classify(R, E):
#classify the example with the rules and to a class
    if R is "no rule yet":
        return "no_rule_here"
    if isinstance(R, list):
        if len(R) <= 0:
            return "no_rule_here"

        prediction = [r.step(E) for r in R]
        results = set(prediction)
        prediction = list(results.remove("no_output_value") if  "no_output_value" in results else results)
    else:
        prediction = R.step(E)

    return prediction

def covered(rule, example, c, debug=False):
#check if the example is covered by the rule
#try to classify with the rule?
    result = classify(rule, example)

    if debug:
        print("checking cover [expected result == given result]: {0} == {1}".format(c, result))

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

def filter(antecendents, ant_to_null, debug=False):
    #from a list of antecendents, filter the desired antecendent
    #return the modified copy of the tree
    hold_idx = -1
    copyTree = antecendents[-2][1].copy_tree()
    copy_ant = copyTree.getAntecedent()

    #search for the first matched antecendent to remove
    for idx, ant in enumerate(antecendents):
        premisse = ant[2]
        if premisse == ant_to_null:
            hold_idx = idx
            side = ant[0]
            break

    if hold_idx == -1:
        if debug:
            print("antecendente não encontrado")
        return copyTree

    origin = copy_ant[hold_idx][1]

    if side == 0:
        new_branch = copy_ant[-2][1].rotation45()

    elif side == -1:
        new_branch = origin.left.rotation45()

    elif side == 1:
        new_branch = origin.right.rotation45()

    return copyTree

    #connectar origem da variavel target com o ramo resultante da rotação 45

def conjuntive_rule(Exemplo, previousRule, leaf, debug = False):
    presence_array = [False] * len(Exemplo)
    current_node = previousRule
    hook = None
    leafNode = Node.Node(value=leaf)

    if previousRule is not "no rule yet":
        while not current_node.is_leaf_node():
            judge = current_node.eval(Exemplo)
            if judge:
                presence_array[current_node.featureIndex] = True
                current_node = current_node.right

            else:
                if current_node.left is not None:
                    current_node = current_node.left
                else:
                    hook = current_node
                    break

    resultRule = None
    previousPremisse = None

    for idx, feature_check in enumerate(presence_array):
        if feature_check:
            continue

        partial_premisse = Node.Node(featureIndex = idx, threshold = Exemplo[idx])

        if resultRule is None:
            resultRule = partial_premisse
        else:
            previousPremisse.set_right(partial_premisse)
        previousPremisse = partial_premisse

    if previousPremisse is not None:
        previousPremisse.set_right(leafNode)

    if hook is not None:
        hook.set_left(resultRule)

    return resultRule

def label_code_block(R, E, true_result, debug = False):

    is_covered = covered(R, E, true_result, debug=debug)
    if debug:
        if is_covered:
            print("regra atual cobre exemplo")
        else:
            print("regra atual NÃO cobre exemplo")

    if is_covered:
        return R

    r = conjuntive_rule(E, R, true_result, debug=debug)

    if debug:
        if r is not None:
            print("conjuntive rule made for %s:"%(true_result))
        else:
            print("conjuntive rule not made")

    if R is None:
        if debug:
            print("new initial rule made for %s:"%(true_result))
        return r

    if r is None:
        return R

    ant_r = r.getAntecedent()

    if debug:
       print("number of antecendents: %d" % (len(ant_r)))

    return r

def Rule_extraction_learning_3(M, C, Ex, theta = 0, debug = False):
    R = dict() 
    for c in C:
        R[c] = "no rule yet"

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
            print("class: {}".format(C[idx]))
            print(c)

    #critério de parada escolhido: repete com o numero de labels e gere [n-ésima volta * quantidade total de labels]
    while voltas < numClasses:
        if debug:
            print("numero de voltas: %d" % (voltas))
        voltas += 1
        qtd_exemplos = numClasses * numClasses * voltas
        E = make_examples(Possibilities, n = qtd_exemplos)

        O = []
        Sum_IO = []
        for example in E:
            model_result = M.predict(np.squeeze(example))
            inputToOutput = M.get_params()["Z"+str(outputLayerIndex)]

            Sum_IO.append(inputToOutput[np.argmax(model_result)])
            output_ONeuron = np.argmax(model_result)
            O.append((output_ONeuron, C[output_ONeuron]))

        if debug:
            print("exemplos gerados: %d" % (len(E)))

        for idx, s in enumerate(Sum_IO):
            for neuron in s:
                ModelOutput =  O[idx][1]
                #if debug:
                #    print("{0} > {1}".format(neuron, theta))
                if neuron > theta:
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
                                R[O[idx][1]] = label_code_block(R[O[idx][1]], E[idx], O[idx][1], debug=debug)

    return R

def parseRules(ruleSet, inputValues):
    resultBatch = []
    for ruleSet in classRuleSets:
        for rule in ruleSet:
            resultBatch.append(rule.step(inputValues))

        resultBatch = set(resultBatch)
        resultBatch = resultBatch.remove("no_output_values") if "no_output_values" in resultBatch else resultBatch
        resultBatch = list(resultBatch)

    return resultBatch if len(resultBatch) > 0 else ["no_results"]

def isComplete(RELruleSet):
    for classLabel, classRules in RELruleSet.items():
        if classRules is "no rule yet":
            return False
    return True

def delete(RELruleSet):
    RELruleSet.clear()