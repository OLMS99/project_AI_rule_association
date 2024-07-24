import random
import math
import numpy as np
import time
from Node import Node
class NodeMofN(Node):
    def __init__(self, featureIndex=None, layerIndex=None, lista=None, threshold=0, comparison="=", left=None, right=None, value="no_input_value", negation = False):
        self.set_comparisons = []

        for item in lista:
            self.set_comparisons.append(item)

        super().__init__(featureIndex=featureIndex, layerIndex=layerIndex, threshold = threshold, comparison = comparison, left = left, right = right, value = value, negation = negation)

    def eval(self, value):
        result = False

        for item in self.set_comparisons:
            M = item[0]
            N = item[1]
            for feature, input_val in zip(N, value):
                count = 0
                count = count + 1 if feature == input_val else 0

            result  = result or count >= N

        return result

    def print(self):

        message =  "avaliação MofN: \n"

        if self.set_comparisons is not None:
            for MofN in self.set_comparisons:
                message += "%s of (%s) \n" % (MofN[1], MofN[0])

        print(message)

        if self.right:
            print("right branch:")
            self.right.print()

        if self.left:
            print("left branch:")
            self.right.print()