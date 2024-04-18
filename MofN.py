import Node

import random
import math
import numpy as np
import time

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
    cluster_members = dict([(i, [networkUnitWeights[i]]) for i in range(nSamples)])

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


        cluster_members[i + nSamples] = cluster_members[x] + cluster_members[y]

        cluster_members[x] = [float('inf')]
        cluster_members[y] = [float('inf')]

    return Z

def influence(clusteredNetwork, cluster, value):
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

def getPossibleClusters(clusteredNetwork, cluster, minValue, maxValue, result=[]):

    if cluster[2] != float('inf') and cluster[4] >= minValue and cluster[4] <= maxValue:
        result.append(cluster)

        #get cluster1 and cluster2

        cluster1 = clusteredNetwork[np.where(clusteredNetwork[:, 0] == cluster[0] and clusteredNetwork[:, 2] != float('inf'))]
        cluster2 = clusteredNetwork[np.where(clusteredNetwork[:, 1] == cluster[1] and clusteredNetwork[:, 2] != float('inf'))]

        if len(cluster1) > 0:
            getPossibleClusters(clusteredNetwork, cluster1[0], minValue, maxValue, result=result)
        if len(cluster2) > 0:
            getPossibleClusters(clusteredNetwork, cluster2[0], minValue, maxValue, result=result)

    return result

def search_set_Au(clusteredNetwork, cluster):
    maxValue = cluster[3]

    possible_values = getPossibleClusters(clusteredNetwork, cluster, 0, maxValue)

    return possible_values

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

#todo: modificar o backwards_propagation para otimizar o bias das unidades
#faça mais dos parametros para NeuralNetwork para fazer a atualização dos pesos e das bias opcional
#reescreva esta para chamá-la e modifica os bias de U com o dicionário change
def optimize(U, model, DataX, Datay, debug = False):


    model.train(DataX[0], Datay[0], DataX[1], Datay[1], update_weights = False, debug = debug)
    params = model.get_params()

    for layer_idx, layer in enumerate(U):
        for neuron_idx, neuron in enumerate(layer):
            neuron[1] = params["b"+str(layer_idx+1)][neuron_idx]

def MofN_2(U, model, DataX, Datay, theta=0, debug=False):
    R=[]
    K=dict()
    G=dict()

    for layer_index, layer in enumerate(U):
        for unit_index, u in enumerate(layer):
            G[unit_index]= cluster_algorithm(u[0])
            K[unit_index]= []
            threshold = u[1] #bias

            print("dados do neuronio: %s" % (u))
            for gi in G[unit_index]:
                print("gi antes de remover: %s" % (gi))
                if not influence(G[unit_index], gi, threshold):
                    remove(G[unit_index], gi)

                    params = model.get_params()
                    weight_idx = getweightIndexes(G[unit_index], len(u[0]))
                    for w in weight_idx:
                        params["W"+str(layer_index+1)][unit_index, int(w):int(w+1)] = 0.0
                    model.load_params(params)

            for gi in G[unit_index]:
                if gi[3] == 0:
                    ki = gi[4]
                else:
                    ki = (gi[4]/gi[3])
                print("gi depois de remover: %s" % (gi))
                print("ki: %s" % (ki))
                K[unit_index].append(ki)#or

    #Handing the constant G, using back propagation algorithm to optimize the bias of u to Ou
    optimize(U, model, DataX, Datay)

    for layer_idx, layer in enumerate(U):
        for u_idx, u in enumerate(layer):
            for gi in G[u_idx]:
                Au = search_set_Au(G[u_idx], gi)
                print("Au: %s" % (Au))
                print("Ku: %s" % (K[u_idx]))

                if len(Au) == 0:
                    Au = [0]*len(K[u_idx])
                print("Au: %s" % (Au))
                print("Ku: %s" % (K[u_idx]))

                if np.asarray(Au).dot(K[u_idx]) > u[1]:
                    for au in Au:
                        leaf_value = u
                        tree = gen_tree([layer_idx, u_idx],au, G[u_idx], leaf_value)
                        R.append(tree)#or

    return R