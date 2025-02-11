import Node
import NeuralNetwork as ANN
import numpy as np
import random
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

#TODO Corrigir bug em que sum(np.squeeze(IO_valor)) >= theta está dando resultados errados
def make_examples(possibilities, Model, theta, n = 1):
    result = []
    tamResult = 0
    outputLayerIndex = Model.get_params()["num layers"] - 1
    for i in range(n):
        oneSample = []

        for e in possibilities.values():
            indexRange = len(e)
            choice = random.randrange(indexRange)
            oneSample.append(e[choice])

        model_result = Model.predict(np.array(oneSample))
        IO_valor = Model.get_params()["A"+str(outputLayerIndex)]

        print("valor de saida: ", np.squeeze(IO_valor))
        print("theta: ", theta)
        print("%s >= %s" % (sum(np.squeeze(IO_valor)), theta))
        if sum(np.squeeze(IO_valor)) >= theta:
            result.append(deepcopy(np.array(oneSample)))
            tamresult = len(result)
            print("input: ", oneSample)
            print("current number of examples generated: ", tamResult)
            continue

        for j, othervalues in enumerate(possibilities.values()):
            for v in othervalues:
                temp = oneSample
                temp[j] = v
                #changing the value of ei to vij increase s?
                modelResult = Model.predict(np.array(temp))
                newSum = Model.get_params()["A"+str(outputLayerIndex)]
                if  sum(np.squeeze(newSum)) > sum(np.squeeze(IO_valor)):
                    oneSample[j] = v
                    IO_valor = newSum

                print("valor de saida: ", np.squeeze(IO_valor))
                print("theta: ", theta)
                print("%s >= %s" % (sum(np.squeeze(IO_valor)), theta))
                if sum(np.squeeze(IO_valor)) >= theta:
                    result.append(deepcopy(np.array(oneSample)))
                    tamResult = len(result)
                    print("input: ", oneSample)
                    print("current number of examples generated: ", tamResult)
                    continue

    return result

def Subset(classNN, rules, examples, debug=False):
#try to classify with the ruleset and compare with class labe
#check if the example is covered by the rulel
    if not isinstance(rules, list):
        return False
    if rules.length() <= 0:
        return False

    result = (rules[0] is not "no rule yet")
    for example in examples:
        ruleLabel = classify(rules, example)
        if debug:
            print("checking cover [expected result == given result]: {0} == {1}".format(c, instanceClass))
        result = result and (ruleLabel == classNN)
    return result

def conjuntive_rule(members, Exemplo, previousRule, leafValue, debug = False):
    raiz = None
    noAnterior = None
    if debug:
        print("criando regra conjuntiva para %s:" % (Exemplo))

    for idx, feature in enumerate(Exemplo):

        novoNo = Node.Node(featureIndex=idx, threshold=feature)

        if raiz is None:
            raiz = novoNo
        else:
            noAnterior.set_right(novoNo)
        noAnterior = novoNo

    if previousRule is not "no rule yet":
        currentNode = raiz
        for i in range(len(Exemplo)):
            try:
                currentNode.append_left(previousRule)
            except:
                previousNode = currentNode
                currentNode = currentNode.right
                previousNode.set_right(None)
                del previousNode
                continue

    noAnterior.set_right(Node.Node(value = leafValue))
    return raiz

def label_code_block(R, members, E, true_result, debug = False):

    is_covered = Subset(true_result, R, [E], debug=debug)
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

    if len(members) <= 4:
        return r

    for key,ant in ant_r.items():
        print(key)
        r_, ant_= r.copy_tree_n_node((key,ant))
        r_ = r_.filter(ant_, debug=debug)
        print("copia modificada da regra feita")
        if Subset(true_result, r_, members, debug=debug):
            print("antecedente retirado")
            r = r_
            ant_r = r.getAntecedent()
            if debug:
                print("number of antecendents after pruning a antecedent: %d" % (len(ant_r)))

    if debug:
       print("number of antecendents after the pruning session: %d" % (len(ant_r)))
    return r

def Rule_extraction_learning_3(M, C, Ex, theta = 0.0, debug = False):
    R = dict()
    for c in C:
        R[c] = "no rule yet"

    Possibilities = possible_values(Ex)
    numClasses = len(C)

    modelParams = M.get_params()
    outputLayerIndex = modelParams["num layers"] - 1

    voltas = 0
    numMaxvoltas = numClasses * numClasses
    exemplosBackup = [[] for _ in C]
    if debug:
        print("Iniciou regras e possibilidades")
        print("numero de labels: %d" % (numClasses))
        for idx, c in R.items():
            print("label: {}".format(idx))
            print("class: {}".format(C[idx]))
            print(c)

    #critério de parada escolhido: repete com o numero de labels e gere [n-ésima volta * quantidade total de labels]
    while voltas < numMaxvoltas:
        if debug:
            print("numero de voltas: %d" % (voltas))
        voltas += 1
        qtd_exemplos = numClasses * numClasses * voltas * voltas
        if debug:
            print("numero de exemplos a serem gerados: %d" % (qtd_exemplos))
        E = make_examples(Possibilities, M, theta, n = qtd_exemplos)

        if debug:
            print("exemplos gerados: %d" % (len(E)))
            #for idx in range(len(E)):
            #    print("LABEL: %s SUM_IO: %s" % (O[idx],Sum_IO[idx]))

        O = []
        Sum_IO = []
        for example in E:
            model_result = M.predict(np.squeeze(example))
            inputToOutput = M.get_params()["A"+str(outputLayerIndex)]

            output_ONeuron = np.argmax(model_result)
            Sum_IO.append(inputToOutput[output_ONeuron][0])
            O.append((output_ONeuron, C[output_ONeuron]))

        for idx, s in enumerate(Sum_IO):
            ModelOutput = O[idx][1]
            exemplosBackup[O[idx][0]].extend(E[idx])
            #print("number os members: %s" % (len(members)))
            #print("current example results %s %s" % (s, ModelOutput))
            R[ModelOutput] = label_code_block(R[ModelOutput], exemplosBackup[O[idx][0]], E[idx], ModelOutput, debug=debug)
            #exemplosBackup[O[idx][0]] = list(set(exemplosBackup[O[idx][0]]))

    return R

def parseRules(classRuleSet, inputValues):
    resultBatch = []
    noOutput = set(["no_output_values"])
    for classification, rule in classRuleSet.items():
        if rule is "no rule yet":
            continue
        resultBatch.extend(rule.step(inputValues))

        resultBatch = set(resultBatch)
        resultBatch = resultBatch - noOutput
        resultBatch = list(resultBatch)

    return resultBatch if len(resultBatch) > 0 else ["no_results"]

def isComplete(RELruleSet):
    if RELruleSet is None:
        return False
    for classLabel, classRules in RELruleSet.items():
        if classRules is "no rule yet":
            return False
    return True

def delete(RELruleSet):
    if RELruleSet is None:
        return
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