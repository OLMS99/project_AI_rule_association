import random
import math
import numpy as np
import time

seed = 1
np.random.seed(seed)

#Node represents a partial premisse or consequent in a rule
#for now it can represent IFthen structure, still need to see how to build efficiently a MofN structure
#layerIndex and feature index -> point to what will be compared
#comparison -> what comparison
#threshold-> what value will compare
#FALSE ->left
#TRUE -> right

#AND-> right
#OR-> left

class Node:
    def __init__(self, featureIndex=None, layerIndex=None, threshold=None, comparison="=", left=None, right=None, value=None, negation = False):

        self.featureIndex = featureIndex
        self.layerIndex = layerIndex
        self.threshold = threshold
        self.comparison = comparison# "=","<",">",">=","<=","!="
        self.negation = negation
        self.left = left
        self.right = right

        self.label = None
        self.value = value

    def num_sons(self):
        return int(self.left is not None) + int(self.right is not None)

    def is_leaf_node(self):
        return (self.value) and (self.left is None) and (self.right is None)

    def set_left(self, node):
        self.left = node

    def set_right(self, node):
        self.right = node

    def probabilityPremise(self, P, B, y, j):
        pass

    def get_node_info(self):
        return (self.negation, self.layerIndex, self.featureIndex, self.threshold, self.comparison, self.value)

    def copy(self):
        copy_node = Node(featureIndex = self.featureIndex,
            layerIndex = self.layerIndex,
            threshold = self.threshold,
            comparison = self.comparison,
            value = self.value,
            negation = self.negation)

        copy_node.set_left(self.left.copy())
        copy_node.set_right(self.right.copy())

        return copy_node

    def step(self, input_values):

        if self.is_leaf_node:
            return self.value

        if comparison == "=":
            if self.layerIndex is None:
                initial_pass = input_values[self.featureIndex] == self.threshold
            else:
                initial_pass = input_values[self.layerIndex][self.featureIndex] == self.threshold

        elif comparison == ">":
            if self.layerIndex is None:
                initial_pass = input_values[self.featureIndex] > self.threshold
            else:
                initial_pass = input_values[self.layerIndex][self.featureIndex] > self.threshold

        elif comparison == "<":
            if self.layerIndex is None:
                initial_pass = input_values[self.featureIndex] < self.threshold
            else:
                initial_pass = input_values[self.layerIndex][self.featureIndex] < self.threshold

        elif comparison == ">=":
            if self.layerIndex is None:
                initial_pass = input_values[self.featureIndex] >= self.threshold
            else:
                initial_pass = input_values[self.layerIndex][self.featureIndex] >= self.threshold

        elif comparison == "<=":
            if self.layerIndex is None:
                initial_pass = input_values[self.featureIndex] <= self.threshold
            else:
                initial_pass = input_values[self.layerIndex][self.featureIndex] <= self.threshold

        elif comparison == "!=":
            if self.layerIndex is None:
                initial_pass = input_values[self.featureIndex] != self.threshold
            else:
                initial_pass = input_values[self.layerIndex][self.featureIndex] != self.threshold

        if negation:
            initial_pass = not initial_pass

        if initial_pass:
            if self.right is not None:
                return self.right.step(input_values)
            else:
                return None

        else:
            if self.left is not None:
                return self.left.step(input_values)
            else:
                return None

    def getAntecedent(self, side = 0, origin = None, archive = []):
        print("entrou na função antecendente")
        self.print()
        if self.is_leaf_node:
            print("nó folha")
            return archive

        parcial_premisse = self.get_node_info()
        OR_branch = None
        AND_branch = None
        print("vendo ramos")

        if self.left:
            print("ramo esquerdo")
            OR_branch = self.left
            archive = self.left.getAntecendent(side = -1, origin = self, archive = archive)

        if self.right:
            print("ramo direito")
            AND_branch = self.right
            archive = self.right.getAntecendent(side = 1, origin = self, archive = archive)

        premisse = [side, origin, parcial_premisse, OR_branch, AND_branch]
        archive.append(premisse)
        print("novo tamanho da lista: %d; entrada da lista: %s" % (len(archive), premisse))
        return archive

    def getHeight(self):
        if self.right is None and self.left is None:
            return 0

        if self.right is not None:
            rightHeight = self.right.getHeight()
        else:
            rightHeight = 0

        if self.left is not None:
            leftHeight = self.left.getHeight()
        else:
            leftHeight = 0

        result = max(rightHeight, leftHeight)

        return result + 1

    def getBalance(self):
        if self.left is None:
            leftHeight = -1
        else:
            leftHeight = self.left.getHeight()

        if self.right is None:
            rightHeight = -1
        else:
            rightHeight = self.right.getHeight()

        return leftHeight - rightHeight

    def getConsequent(self, consequent=[]):
        if self.is_leaf_node:
            consequent.append(self.value)
        else:
            if self.right is not None:
                consequent.append(['R', self.right.getConsequent(consequent=consequent)])

            if self.left is not None:
                consequent.append(['L', self.left.getConsequent(consequent=consequent)])

        return consequent

    def print(self):
        if self.is_leaf_node():
            print("folha: %s" % (self.value))
            return

        if self.layerIndex:
            print("neuronio avaliado: camada %d numero %d" % (self.featureIndex, self.featureIndex))
        else:
            print("neuronio avaliado: numero %d" % (self.featureIndex))

        if self.negation:
            print("NOT neuronio %s %s" % (self.comparison, self.threshold))
        else:
            print("neuronio %s %s" % (self.comparison, self.threshold))

        if self.right is not None:
            print("right branch:")
            self.right.print()

        if self.left is not None:
            print("left branch:")
            self.right.print()

    def rotation45():
    #é assumido que os ramos imediatos não são nulos
    #verifica os nós esquerda e direita, quantos subnós cada um tem, de que lado está conectado
    #primeiro verifica o nós direito e veja os sub nós, quais são folhas

        if self.right is None:
            return self

        if self.left is None:
            return self

        leftLeaf = self.left.is_leaf_node()
        rightLeaf = self.right.is_leaf_node()

        if leftLeaf and rightLeaf:
            return

        elif leftLeaf:
            self.right.set_left(leftLeaf)
            return self.right

        elif rightLeaf:
            self.left.set_right(rightLeaf)
            return self.left

        leftSons = (self.left.num_sons(), selt.left.left, selt.left.right)
        rightSons = (self.right.num_sons(), selt.right.left, selt.right.right)

        if leftSons[0] == 0:
            self.left.set_right(self.right)
            return self.left

        elif rightSons[0] == 0:
            self.right.set_left(self.left)
            return self.right

        elif leftSons[0] == 1:
            if leftSons[1]:
                self.left.set_right(self.right)
            elif leftSons[2]:
                self.left.set_left(self.right)
            return self.left

        elif rightSons[0] == 1:
            if rightSons[1]:
                self.right.set_rightt(self.left)
            elif rightSons[2]:
                self.right.set_left(self.left)
            return self.right


        elif leftSons[0] == 2:

            resultNode = self.left.rotation45()

            if resultNode is None:

                resultNode = self.right.rotation45()

                if resultNode.left:
                    self.right.set_left(None)
                    self.right.set_right(resultNode)

                elif resultNode.right:
                    self.right.set_left(resultNode)
                    self.right.set_right(None)

                return self.right

            if resultNode.left:
                self.right.set_left(None)
                self.right.set_right(resultNode)

            elif resultNode.right:
                self.right.set_left(resultNode)
                self.right.set_right(None)

            return self.right