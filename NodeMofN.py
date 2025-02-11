import random
import math
import numpy as np
from copy import deepcopy
import time
from collections import deque
from Node import Node, DontUse
class NodeMofN(Node):
    #TODO rever como set comparisons está implementado se deveria ser de outra forma
    def __init__(self, featureIndex = DontUse, layerIndex = DontUse, listaPremissas = None, threshold = 0, comparison = "=", left = None, right = None, value = "no_input_value", negation = False):
        self.set_comparisons = []

        for item in listaPremissas:
            self.set_comparisons.append(item)

        super(NodeMofN, self).__init__(featureIndex = featureIndex, layerIndex = layerIndex, threshold = threshold, comparison = comparison, left = left, right = right, value = value, negation = negation)

    def __hash__(self):
        values_comparisons_sets = set()
        indexes_comparisons_set = set()
        for item in self.set_comparisons:
            values_comparisons_sets.add(item[1])
            if len(item[0]) == 2:
                indexes_comparisons_set.add(item[0][0])
                indexes_comparisons_set.add(item[0][1])
            if len(item[0]) == 1:
                indexes_comparisons_set.add(item[0])

        return hash((self.featureIndex, self.layerIndex, self.threshold, tuple(indexes_comparisons_set), tuple(values_comparisons_sets), self.negation, self.value, self.fabricationTime))

    def eval(self, value):
        result = False
        count = 0
        for item in self.set_comparisons:
            coord = item[0]
            N = item[1]
            count = count + 1 if value[coord[0]][coord[1]] == N else count

        if self.comparison not in ("=","==","!=",">","<",">=","<="):
            raise Exception("Not valid comparison, has been set %s" % (self.comparison))

        if self.comparison == "=" or self.comparison == "==":
            result  = result or count == self.threshold
        if self.comparison == "!=":
            result  = result or count != self.threshold
        if self.comparison == ">":
            result  = result or count > self.threshold
        if self.comparison == "<":
            result  = result or count < self.threshold
        if self.comparison == ">=":
            result  = result or count >= self.threshold
        if self.comparison == "<=":
            result  = result or count <= self.threshold

        return result

    def step(self, input_values):
        if self.featureIndex is DontUse and self.layerIndex is DontUse:
            return
        if self.is_leaf_node():
            return self.value

        initial_pass = self.eval(input_values)

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

    def equal(self, node):
        if node is None:
            return False

        if not isinstance(node,NodeMofN):
            return False

        compMN = True
        for itemA, itemB in zip(self.set_comparisons, node.set_comparisons):
            compMN = compMN and ((itemA[0][0] == itemB[0][0] and itemA[0][1] == itemB[0][1]) and itemA[1] == itemB[1])
        comp1 = self.label == node.label
        comp2 = self.comparison == node.comparison
        comp3 = self.negation == node.negation
        comp4 = self.threshold == node.threshold
        comp5 = self.value == node.value
        return compMN and comp1 and comp2 and comp3 and comp4 and comp5

    def copy_node(self):
        return NodeMofN(featureIndex = deepcopy(self.featureIndex),
        layerIndex = deepcopy(self.layerIndex),
        listaPremissas = deepcopy(self.set_comparisons),
        threshold = deepcopy(self.threshold),
        comparison = deepcopy(self.comparison),
        value = deepcopy(self.value),
        negation = deepcopy(self.negation))

    def print(self):

        message =  "avaliação MofN: \n"
        message += "threshold(M) = %d \n" % (self.threshold)
        if self.set_comparisons is not None:
            for MofN in self.set_comparisons:
                message += "(%s, %s) \n" % ( MofN[1], MofN[0])

        print(message)

        if self.right:
            print("right branch:")
            self.right.print()

        if self.left:
            print("left branch:")
            self.right.print()

    def destroy(self):
        for premissa in self.set_comparisons:
            del premissa
        super().destroy()