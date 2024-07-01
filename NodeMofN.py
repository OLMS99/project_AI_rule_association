import random
import math
import numpy as np
import time
from Node import Node
class NodeMofN(Node):
    def __init__(self, lista=None, threshold=0, comparison="=", left=None, right=None, value="no_input_value", negation = False):
        self.set_comparisons = dict()
        if lista is None:
            lista = []

        for item in lista:  #ai -> valor de criterios mínimos
                            #-> premissas (pegar os indices para comparar)
            if not self.set_comparisons[item[0]]:
                self.set_comparisons[item[0]] = set(item[1])
            else:
                self.set_comparisons[item[0]].update(item[1])

        super().__init__(threshold = threshold, comparison = comparison, left = left, right = right, value = value, negation = negation)

    def eval(self, value):
        count = 0

        for idx, val in value:
            if isinstance(val, list):
                for feature_idx, feature_val in val:
                    if feature_val in self.set_comparisons[(idx,feature_idx)]:
                        count+=1
            else:
                if val in self.set_comparisons[(idx)]:
                    count+=1

        if comparison == "=":
            initial_pass = count == self.threshold

        elif comparison == ">":
            initial_pass = count > self.threshold

        elif comparison == "<":
            initial_pass = count < self.threshold

        elif comparison == ">=":
            initial_pass = count >= self.threshold

        elif comparison == "<=":
            initial_pass = count <= self.threshold

        elif comparison == "!=":
            initial_pass = count != self.threshold

        if negation:
            initial_pass = not initial_pass

        return initial_pass

    def print():

        message =  "avaliação MofN: \n"

        if self.negation:
            message += "NOT "

        message += self.comparison + " " + self.threshold + " do"

        massage += " conjunto de comparações possiveis do nó:"
        for neuron, vals in self.set_comparisons.items():
            message += "\nneuronio {0} = {1}".format(neuron,vals)

        print(message)

        if self.right:
            print("right branch:")
            self.right.print()

        if self.left:
            print("left branch:")
            self.right.print()