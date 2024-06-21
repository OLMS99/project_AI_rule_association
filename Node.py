import random
import math
import numpy as np
import time

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
    def __init__(self, featureIndex=None, layerIndex=None, threshold=None, comparison="=", left=None, right=None, value="no_output_value", negation = False):

        self.featureIndex = featureIndex
        self.layerIndex = layerIndex
        self.threshold = threshold
        self.comparison = comparison# "=","<",">",">=","<=","!=", "=="
        self.negation = negation
        self.left = left
        self.right = right

        self.label = None
        self.value = value

    def getInputNeuron():
        return [self.layerIndex, self.featureIndex]

    def len():
        len_left = self.left.len() if self.left is not None else 0
        len_right = self.right.len() if self.right is not None else 0
        return 1 + len_left + len_right

    def num_sons(self):
        return int(self.left is not None) + int(self.right is not None)

    def set_label(self, label):
        self.label = str(label)

    def eval(self, value):
        if isinstance(value, list):
            if self.layerIndex:
                holder = value[self.layerIndex][self.featureIndex]
            else:
                holder = value[self.featureIndex]
        else:
            holder = value

        if self.comparison == "=" or self.comparison == "==":
            initial_pass = holder == self.threshold

        elif self.comparison == ">":
            initial_pass = holder > self.threshold

        elif self.comparison == "<":
            initial_pass = holder < self.threshold

        elif self.comparison == ">=":
            initial_pass = holder >= self.threshold

        elif self.comparison == "<=":
            initial_pass = holder <= self.threshold

        elif self.comparison == "!=":
            initial_pass = holder != self.threshold

        if self.negation:
            initial_pass = not initial_pass

        return initial_pass

    def is_leaf_node(self, debug=False):
        statement_1 = self.num_sons() == 0
        statement_2 = self.value != "no_output_value"
        if debug:
            print("é um nó folha?")
            print("tem nenhum sub-nó: %s"%(statement_1))
            print("tem um valor diferente de 'no_input_value': %s"%(statement_2))
        return statement_1 and statement_2

    def set_left(self, node):
        self.left = node

    def set_right(self, node):
        self.right = node

    def append_left(self, node):
        if self.left:
            self.left.append_left(node)
        else:
            self.set_left(node)

    def append_right(self, node):
        if self.right:
            self.right.append_right(node)
        else:
            self.set_right(node)

    def probabilityPremise(self, P, B, y, j):
        pass

    def equal_antecedent(self, premisse):
        if premisse[2] == self.featureIndex \
            and premisse[1] == self.layerIndex \
            and premisse[3] == self.threshold \
            and premisse[0] == self.negation \
            and premisse[4] == self.comparison:
            return True
        return False

    def equal_consequent(self, premisse):
        if premisse[5] == self.value:
            return True
        return False

    def equal_premisse(self, premisse):
        if self.equal_antecedent(premisse) or self.equal_consequent(premisse):
            return True
        return False

    def get_node_info(self):
        return (self.negation, self.layerIndex, self.featureIndex, self.threshold, self.comparison, self.value, self.label)

    def copy_node(self):
        return Node(featureIndex = self.featureIndex,
            layerIndex = self.layerIndex,
            threshold = self.threshold,
            comparison = self.comparison,
            value = self.value,
            negation = self.negation)

    def copy_tree(self):
        copy_node = self.copy_node()

        if self.left:
            copy_node.set_left(self.left.copy_tree())

        if self.right:
            copy_node.set_right(self.right.copy_tree())

        return copy_node

    def step(self, input_values):

        if self.is_leaf_node():
            return self.value

        if self.layerIndex:
            initial_pass = self.eval(input_values[self.layerIndex][self.featureIndex])
        else:
            initial_pass = self.eval(input_values[self.featureIndex])

        if initial_pass:
            if self.right:
                return self.right.step(input_values)
            else:
                return "no_output_value"

        else:
            if self.left:
                return self.left.step(input_values)
            else:
                return "no_output_value"

        return "no_output_value"

    def getAntecedent(self, side = 0, origin = None, debug = False):

        archive = []

        if self.is_leaf_node():
            if debug:
                print("nó folha")
            return archive

        parcial_premisse = self.get_node_info()
        OR_branch = None
        AND_branch = None

        if debug:
            print("vendo ramos")

        if self.left:
            if debug:
                print("ramo esquerdo")
            OR_branch = self.left
            archive.extend(self.left.getAntecedent(side = -1, origin = self, debug=debug))

        if self.right:
            if debug:
                print("ramo direito")
            AND_branch = self.right
            archive.extend(self.right.getAntecedent(side = 1, origin = self, debug=debug))

        premisse = [side, origin, parcial_premisse, OR_branch, AND_branch]
        archive.append(premisse)
        if debug:
            print("novo tamanho da lista: %d; entrada da lista: %s" % (len(archive), premisse))

        return archive

    def getHeight(self):
        if self.right is None and self.left is None:
            return 0

        if self.right:
            rightHeight = self.right.getHeight()
        else:
            rightHeight = 0

        if self.left:
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
            if self.right:
                consequent.append(['R', self.right.getConsequent(consequent=consequent)])

            if self.left:
                consequent.append(['L', self.left.getConsequent(consequent=consequent)])

        return consequent

    def print(self):
        if self.is_leaf_node():
            print("folha: ", self.value)
            return

        message =  "neuronio avaliado: "

        if self.negation:
            message += "NOT "

        if self.layerIndex:
            message += str("camada: {}").format(self.layerIndex)

        if self.featureIndex:
            message += str(" neuronio: {}").format(self.featureIndex)
            message += str("\nneuronio {0} {1}").format(self.comparison, str(self.threshold))

        print(message)

        if self.right:
            print("right branch:")
            self.right.print()

        if self.left:
            print("left branch:")
            self.right.print()

    def rotation45(self):
    #verifica os nós esquerda e direita, quantos subnós cada um tem, de que lado está conectado
    #primeiro verifica o nós direito e veja os sub nós, quais são folhas

        if self.right is None and self.left is None:
            return None

        elif self.right is None:
            return self.left

        elif self.left is None:
            return self.right

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

        leftSons = (self.left.num_sons(), self.left.left, self.left.right)
        rightSons = (self.right.num_sons(), self.right.left, self.right.right)

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