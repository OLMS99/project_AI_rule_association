import Node
import NodeMofN

import numpy as np

def distance(data1, data2):
    return abs(sum(data1) - sum(data2))

def argmin(cases):
    minx,miny = (0,0)
    min_val = float("inf")
    for i in range(cases.shape[0]):
        for j in range(cases.shape[0]):
            if j==i:
                continue

            if cases[i, j] < min_val:
                min_val = cases[i, j]
                minx = i
                miny = j

    return min_val, minx, miny

def cluster_distance(cluster_members):
    nClusters = len(cluster_members)
    keys = list(cluster_members.keys())
    Distance = np.zeros((nClusters, nClusters))

    for i in range(nClusters):
        ith_elems = cluster_members[keys[i]]
        for j in range(nClusters):

            jth_elems = cluster_members[keys[j]]
            d_in_clusters = distance(ith_elems, jth_elems)
            dij = np.min(d_in_clusters)
            Distance[i,j] = dij

    return Distance

def cluster_algorithm(networkUnitWeights):
    nSamples = len(networkUnitWeights)
    cluster_members = {i : [networkUnitWeights[i]] for i in range(nSamples)}

    Z = np.zeros(shape = (nSamples-1, 5)) #c1, c2, distance, count, sum

    for i in range(0, nSamples-1):
        nClusters = len(cluster_members)
        keys = list(cluster_members.keys())

        d = cluster_distance(cluster_members)
        _, minx, miny = argmin(d)

        x = keys[minx]
        y = keys[miny]

        Z[i, 0] = x
        Z[i, 1] = y
        Z[i, 2] = d[x, y]
        Z[i, 3] = len(cluster_members[x]) + len(cluster_members[y])
        Z[i, 4] = sum(cluster_members[x]) + sum(cluster_members[y])

        novaLista = []
        novaLista.extend(cluster_members[x])
        novaLista.extend(cluster_members[y])
        cluster_members[i + nSamples] = novaLista

        cluster_members[x] = [float('inf')]
        cluster_members[y] = [float('inf')]

    return Z

def influence(cluster, value):
    #clusterNetwork, matrix cada linha tem cluster1, cluster2, distancia, tamanho do super cluster, soma dos valores do super cluster
    #cluster, peso(s) sendo avaliado, uma linha da matrix
    #value, threshold, bias do neuronio que dos pesos originou o cluster analisado

    #a influencia do cluster é assumida ser a soma dos valores do cluster ser igual ou maior que um valor

    return cluster[4] >= value

def remove(clusterNetwork, cluster):
    searched = cluster

    #procura em cada linha, quando encontrado troque a linha por um valor invalido e atualize a próxima linha que tem a chave do cluster
    #nas linha subsequentes, atualize o valor das colunas 3 e 4 para atualizar o tamanho e a soma
    #repita a etapa anterior até a ultima linha do clusterNetwork

    for i, linha in enumerate(clusterNetwork):
        if all(linha == searched):

            clusterNetwork[i, 2] = float('inf')
            clusterNetwork[i, 3] = 0
            clusterNetwork[i, 4] = 0

def getweightIndexes(clusterNetwork, numWeights):
    weightIndexes = []
    for line in clusterNetwork:
        if line[4] == 0:
            if line[0] < numWeights:
                weightIndexes.append(line[0])

            if line[1] < numWeights:
                weightIndexes.append(line[1])

    return weightIndexes

def getPossibleClusters(clusteredNetwork, minValue, maxValue):
    mascara = (clusteredNetwork[:,2]!=float('inf')) & (clusteredNetwork[:,4] >= minValue) & (clusteredNetwork[:,4] <= maxValue)
    result = clusteredNetwork[mascara]
    return result

def search_set_Au(clusternetwork, clusterValue):
    giSize = clusterValue[3]
    AuSetsMask = (clusternetwork[:,3] <= giSize)
    return clusternetwork[AuSetsMask][:,3]

def makerule(layer_idx, neuron_idx, val, premisses, leaf_value):
    #newRule = gen_tree(neuron_idx, val, premisses, leaf_value)
    #premisses -> gi
    #val -> ai

    newRule = NodeMofN.NodeMofN(featureIndex = neuron_idx, layerIndex = layer_idx, lista=[[premisses, val]], comparison="=", negation = False)
    folha = Node.Node(value = leaf_value)
    newRule.append_right(folha)
    return newRule

#caso precise de uma arvore de node convencional, esta função gera regras sem nodes MofN
def gen_tree(neuron_idx, val, premisses, leaf_value, counter=0, idx=0):
    if idx < len(premisses):
        cur_node = Node.Node(featureIndex = neuron_idx[1], layerIndex = neuron_idx[0], threshold=val, comparison="=", negation = False)
        cur_node.set_left(gen_tree(neuron_idx, val, premisses, leaf_value, counter=counter, idx=idx+1))
        cur_node.set_right(gen_tree(neuron_idx, val, premisses, leaf_value, counter=counter+1, idx=idx+1))
        return cur_node

    else:
        if val == counter:
            cur_node = Node.Node(value = leaf_value)
        else:
            cur_node = Node.Node(value = val !=counter)

        return cur_node

def optimize(U, model, DataX, Datay, debug = False):

    model.train(DataX[0], Datay[0], DataX[1], Datay[1], epochs=100, update_weights = False, debug = debug)
    params = model.get_params()

    for layer_idx, layer in enumerate(U):
        for neuron_idx, neuron in enumerate(layer):
            neuron[1] = params["b"+str(layer_idx+1)][neuron_idx]

def MofN_2(U, model, DataX, Datay, theta=0, debug=False):
    R = []

    K = dict()
    G = dict()
    A = dict()

    Backup = dict()

    for layer_index, layer in enumerate(U):
        for unit_index, u in enumerate(layer):
            neuron_coord = (layer_index, unit_index)
            G[neuron_coord] = cluster_algorithm(u[0]) #weights
            Backup[neuron_coord] = G[neuron_coord].copy()
            K[neuron_coord] = []
            A[neuron_coord] = []
            threshold = u[1] #bias

            if debug:
                print("dados do neuronio: %s" % (u))
            for gi in G[neuron_coord]:
                if debug:
                    print("gi antes de remover: %s" % (gi))
                if not influence(gi, threshold):
                    remove(G[neuron_coord], gi)

                    params = model.get_params()
                    weight_idx = getweightIndexes(G[neuron_coord], len(u[0]))
                    for w in weight_idx:

                        params["W"+str(layer_index+1)][unit_index, int(w)] = 0.0
                    model.load_params(params)

            #limpeza
            G[neuron_coord] = G[neuron_coord][G[neuron_coord][:,2]!=float('inf')]

            for gi in G[neuron_coord]:
                razao = 1 if gi[3] == 0 else gi[3]
                ki = gi[4]/razao
                if debug:
                    print("gi depois de remover: %s" % (gi))
                Au = search_set_Au(G[neuron_coord], gi)
                if len(Au) == 0:
                    Au = [0]*len(K[neuron_coord])
                if debug:
                    print("Au: %s" % (Au))
                    print("Ku: %s" % (K[neuron_coord]))
                A[neuron_coord].append(Au)
                K[neuron_coord].append(ki)#or

    #Handing the constant G, using back propagation algorithm to optimize the bias of u to Ou
    optimize(U, model, DataX, Datay)

    for layer_idx, layer in enumerate(U):
        layerRules = []
        for u_idx, u in enumerate(layer):
            neuron_coord = (layer_idx, u_idx)
            for gi in G[neuron_coord]:
                if debug:
                    print("Au dot Ku: %s . %s" % (Au, K[neuron_coord]))
                    print("Ou: %s" % (u[1]))
                for ai in Au:
                    if np.asarray(ai).dot(K[neuron_coord])[0] > u[1]:
                        newRule = makerule(layer_idx, u_idx, np.asarray(ai), gi, (layer_idx + 1, u_idx))
                        layerRules.append(newRule)#or

        R.append(layerRules)

    return R

def parseRules(ruleSet, model, inputValues):
    model.predict(inputValues)
    model_values = model.getAtributes()
    results = []
    for layerRules in ruleSet:
        currResults = []
        for rule in layerRules:
            currResults.append(rule.step(model_values))

        currResults = set(currResults)
        currResults = currResults.remove("no_output_values") if "no_output_values" in currResults else currResults
        results = list(currResults)

    return results if len(results) > 0 else ["no_results"]

def isComplete(MofNruleSet):
    for layerRules in MofNruleSet:
        if len(layerRules) <= 0:
            return False
    return True