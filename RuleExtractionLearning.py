import Node
import NeuralNetwork as ANN
import numpy as np
import pandas as pd
import gc
from copy import deepcopy

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

def covered(rule, instances, c, debug=False):
#check if the example is covered by the rule
#try to classify with the rule?
    result = True
    for example in instances:
        instanceClass = classify(rule, example)
        if debug:
            print("checking cover [expected result == given result]: {0} == {1}".format(c, instanceClass))
        result = result and (instanceClass == c)

    if result == "no_rule_here":
        return False

    return result

def possible_values(examplesData, database = None, debug = False):
#valores possiveis para os exemplos
    if database is not None:
        result = deepcopy(database)
    else:
        result = dict()
    #print("dimensões: %s" % (examplesData.shape))

    feature_length = examplesData.shape[1]
    if debug:
        print("numero de exemplos: %s" % (examplesData.shape[0]))

    for i in range(feature_length):
        result[i] = []

    for e in examplesData:
        for i in range(feature_length):
            result[i].append(e[i])
    if debug:
        for idx, data in result.items():
            print("tamanho dos dados do neurônio %s: %s" % (idx, len(data)))

    for idx, r in result.items():
        result[idx] = list(set(r))

    if debug:
        for idx, data in result.items():
            print("valores possíveis do neurônio %s: %s" % (idx, len(data)))

    return result

def make_examples(possibilities, Model, theta, n = 1):
    result = []
    outputLayerIndex = Model.get_params()["num layers"] - 1
    for i in range(n):
        oneSample = []

        for e in possibilities.values():
            indexRange = len(e)
            choice = np.random.randint(indexRange)
            oneSample.append(e[choice])

        model_result = Model.predict(np.array(oneSample))
        IO_valor = Model.get_params()["Z"+str(outputLayerIndex)]

        if max(IO_valor) >= theta:
            result.append(deepcopy(np.array(oneSample)))
            continue

        for i, othervalues in enumerate(possibilities.values()):
            for v in othervalues:
                temp = oneSample
                temp[i] = v
                #changing the value of ei to vij increase s?
                modelResult = M.predict(temp)
                newSum = M.get_params()["Z"+str(outputLayerIndex)]

                if  max(newSum) > max(IO_valor):
                    oneSample[i] = v
                    IO_valor = newSum

                if max(IO_valor) >= theta:
                    result.append(deepcopy(np.array(oneSample)))

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

#TODO: refatorar
def Subset(classNN, rules, examples):
#try to classify with the ruleset and compare with class label
    result = True
    for example in examples:
        ruleLabel = classify(rules, example)
        result = result and (ruleLabel == classNN)
    return result
#TODO: refatorar
def conjuntive_rule(members, Exemplo, previousRule, leaf, debug = False):
    presence_array = [False] * len(Exemplo)
    current_node = previousRule
    hook = None
    leafNode = Node.Node(value=leaf)

    if previousRule is not "no rule yet":
        while not current_node.is_leaf_node():
            print(Exemplo)
            judge = current_node.eval(Exemplo)
            print(judge)
            if judge:
                presence_array[current_node.featureIndex] = True
                current_node = current_node.right
                continue

            if current_node.left is None:
                hook = current_node
                break

            current_node = current_node.left

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

def label_code_block(R, members, E, true_result, debug = False):

    is_covered = covered(R, [E], true_result, debug=debug)
    if debug:
        if is_covered:
            print("regra atual cobre exemplo")
        else:
            print("regra atual NÃO cobre exemplo")

    if is_covered:
        return R

    r = conjuntive_rule(members, E, R, true_result, debug=debug)

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
    exemplosBackup = []
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
        qtd_exemplos = numClasses * numClasses * voltas * voltas
        E = make_examples(Possibilities, M, theta, n = qtd_exemplos)
        exemplosBackup.extend(E)

        O = []
        Sum_IO = []
        for example in E:
            model_result = M.predict(np.squeeze(example))
            inputToOutput = M.get_params()["Z"+str(outputLayerIndex)]

            output_ONeuron = np.argmax(model_result)
            Sum_IO.append(inputToOutput[output_ONeuron][0])
            O.append((output_ONeuron, C[output_ONeuron]))

        if debug:
            print("exemplos gerados: %d" % (len(E)))
            for idx in range(len(E)):
                print("LABEL: %s SUM_IO: %s" % (O[idx],Sum_IO[idx]))

        for idx, s in enumerate(Sum_IO):
            members = E[:idx]
            ModelOutput = O[idx][1]
            print("number os members: %s" % (len(members)))
            print("current example results %s %s" % (s, ModelOutput))
            R[ModelOutput] = label_code_block(R[ModelOutput], members, E[idx], ModelOutput, debug=debug)

    return R

def parseRules(classRuleSet, inputValues):
    resultBatch = []
    for classification, rule in classRuleSet.items():
        if rule is None:
            continue
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

def printRules(classRuleSets):
    for label,ruleset in classRuleSets.items():
        if ruleset is not None:
            if ruleset == "no rule yet":
                print("no rule made for %s" % (label))
                continue
            print("rule made for %s" % (label))
            ruleset.print()
        else:
            print("no rule made for %s" % (label))
    print(classRuleSets)