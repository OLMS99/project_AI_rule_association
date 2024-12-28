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

class DontUse: pass
class Node:
    def __init__(self, featureIndex = DontUse, layerIndex= DontUse, threshold=None, comparison="=", left=None, right=None, value="no_output_value", negation = False):

        if featureIndex is None and layerIndex is None and value is "no_output_value":
            raise Exception("Node must have a reference of what neuron will be validade, it cannot have both featureIndex and layerIndex as None")

        self.featureIndex = featureIndex
        self.layerIndex = layerIndex
        self.threshold = threshold
        self.comparison = comparison# "=","<",">",">=","<=","!=", "=="
        self.negation = negation

        if not isinstance(left, Node) and left is not None:
            raise Exception("Node only accept Nodes or derivates as sons, left is a %s" % (type(node)))

        self.left = left if isinstance(left, Node) else None

        if not isinstance(right, Node) and right is not None:
            raise Exception("Node only accept Nodes or derivates as sons, right is a %s" % (type(node)))

        self.right = right if isinstance(right, Node) else None

        self.label = None
        self.value = value
        self.numSons = int(self.left is not None) + int(self.right is not None)
        self.isLeaf = (self.value != "no_output_value") and (self.numSons == 0)

    def getValue(self):
        return self.value

    def getLabel(self):
        return self.label

    def equal(self, node):
        if node is None:
            return False

        comp1 = self.label == node.label
        comp2 = self.featureIndex == node.featureIndex
        comp3 = self.layerIndex == node.layerIndex
        comp4 = self.comparison == node.comparison
        comp5 = self.negation == node.negation
        comp6 = self.threshold == node.threshold
        comp7 = self.value == node.value
        return comp1 and comp2 and comp3 and comp4 and comp5 and comp6 and comp7

    def copy(self):
        esquerda = self.left.copy() if self.left is not None else None
        direita = self.right.copy() if self.right is not None else None
        newNode = Node(featureIndex=self.featureIndex, layerIndex=self.layerIndex, threshold=self.threshold, comparison = self.comparison, left = esquerda, right = direita, value = self.value, negation = self.negation)
        newNode.label = self.label
        return newNode

    def getInputNeuron(self):

        if self.layerIndex is DontUse:
            return [self.featureIndex]
        if self.featureIndex is DontUse:
            return [self.layerIndex]

        return [self.layerIndex, self.featureIndex]

    def getOutputs(self):
        outputs = []
        left = self.left
        right = self.right
        if left is not None:
            if left.is_leaf_node():
                outputs.append(left.value)
            else:
                outputs.extend(left.getOutputs)

        if right is not None:
            if right.is_leaf_node():
                outputs.append(right.value)
            else:
                outputs.extend(right.getOutputs)

        outputs = list(set(outputs)).remove("no_output_value")

        return outputs

    def len(self):
        len_left = self.left.len() if self.left is not None else 0
        len_right = self.right.len() if self.right is not None else 0
        return 1 + len_left + len_right

    def num_sons(self):
        return self.numSons

    def set_label(self, label):
        self.label = str(label)

    def eval(self, value):
        if isinstance(value, list):
            if self.layerIndex is not DontUse and self.featureIndex is not DontUse:
                holder = value[self.layerIndex][self.featureIndex]
            elif self.featureIndex is not DontUse:
                holder = value[self.featureIndex]
            elif self.layerIndex is not DontUse:
                holder = value[self.layerIndex]
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
        if self.isLeaf is None:
            statement_1 = self.numSons == 0
            statement_2 = self.value != "no_output_value"
            self.isLeaf = statement_1 and statement_2
            if debug:
                print("é um nó folha?")
                print("tem nenhum sub-nó: %s"%(statement_1))
                print("tem um valor diferente de 'no_output_value': %s"%(statement_2))

        return self.isLeaf

    def set_left(self, node):
        if not isinstance(node, Node) and node is not None:
            raise Exception("Node only accept Nodes or derivates as sons, tried set left with a %s" % (type(node)))
        if self.equal(node):
            raise Exception("Node doesn't connect to itself")

        self.left = node
        self.numSons = int(self.left is not None) + int(self.right is not None)
        self.isLeaf = (self.value != "no_output_value") and (self.numSons == 0)

    def set_right(self, node):
        if not isinstance(node, Node) and node is not None:
            raise Exception("Node only accept Nodes or derivates as sons, tried set right with a %s" % (type(node)))
        if self.equal(node):
            raise Exception("Node doesn't connect to itself")

        self.right = node
        self.numSons = int(self.left is not None) + int(self.right is not None)
        self.isLeaf = (self.value != "no_output_value") and (self.numSons == 0)

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

    @staticmethod
    def equal_antecedent(premisse1, premisse2):
        #(self.negation, self.layerIndex, self.featureIndex, self.threshold, self.comparison, self.value, self.label)
        if premisse1[0] != premisse2[0]:
            return False
        if not (premisse1[1] is DontUse and premisse2[1] is DontUse) and not (premisse1[1] == premisse2[1]):
            return False
        if not (premisse1[2] is DontUse and premisse2[2] is DontUse) and not (premisse1[2] == premisse2[2]):
            return False
        if premisse1[3] != premisse2[3]:
            return False
        if premisse1[4] != premisse2[4]:
            return False
        if premisse1[5] != premisse2[5]:
            return False
        return True

    def filter(self, ant_to_null, debug=False):
        #from a list of antecendents, filter the desired antecendent
        #return the modified copy of the tree
        hold_idx = -1
        antList = self.getAntecedent()

        #search for the first matched antecendent to remove
        for idx, ant in enumerate(antList):
            premisse = ant[2]
            searchedPremisse = ant_to_null[2]
            if Node.equal_antecedent(premisse, searchedPremisse):
                hold_idx = idx
                side = ant[0]
                origin = ant[1]
                break

        if debug:
            print("antecedentes analizados")
        if hold_idx == -1:
            if debug:
                print("antecendente não encontrado")
            return copyTree

        if debug:
            print("iniciando poda do antecedente")
        if side == 0:
            new_branch = self.rotation45()
            origin = new_branch
        elif side == -1:
            new_branch = origin.left.rotation45()
            origin.set_left(new_branch)
        elif side == 1:
            new_branch = origin.right.rotation45()
            origin.set_right(new_branch)

        if side !=0:
            if debug:
                print("numero de antecedentes da nova arvore: %s" % (len(origin.getAntecedent())))

        return self

        #connectar origem da variavel target com o ramo resultante da rotação 45

    @staticmethod
    def equal_consequent(premisse1, premisse2):
        if premisse1[5] == premisse2[5]:
            return True
        return False

    def equal_premisse(self, premisse):
        return self.equal_antecedent(premisse) or self.equal_consequent(premisse)

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

    def destroy(self):
        if self.left is not None:
            self.left.destroy()

        if self.right is not None:
            self.right.destroy()

        del self.value
        del self.label
        del self

    def step(self, input_values):

        if self.is_leaf_node():
            return self.value

        if self.layerIndex is not DontUse and self.featureIndex is not DontUse:
            initial_pass = self.eval(input_values[self.layerIndex][self.featureIndex])
        elif self.featureIndex is not DontUse:
            initial_pass = self.eval(input_values[self.featureIndex])

        if initial_pass:
            if self.right is not None:
                return self.right.step(input_values)
            else:
                return "no_output_value"

        else:
            if self.left is not None:
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

        if self.left is not None:
            if debug:
                print("ramo esquerdo")
            OR_branch = self.left
            archive.extend(self.left.getAntecedent(side = -1, origin = self, debug=debug))

        if self.right is not None:
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

    def getConsequent(self, consequent=None):
        if consequent is None:
            consequent = []

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

        if self.layerIndex is not DontUse:
            message += str("camada: {}").format(self.layerIndex)

        if self.featureIndex is not DontUse:
            message += str(" neuronio: {}").format(self.featureIndex)
            message += str("\nvalor do neuronio {0} {1}").format(self.comparison, str(self.threshold))

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
            return None

        elif leftLeaf:
            self.right.set_left(self.left)
            return self.right

        elif rightLeaf:
            self.left.set_right(self.right)
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