import random
import math
import numpy as np
import time
import Node

class NodeTree(Node):
    def __init__(self, tree=None, threshold=True, left=None, right=None, value="no_input_value", negation = False):
        self.root = tree

        super().__init__(threshold = threshold, left = left, right = right, value = value, negation = negation)

    def eval(self, value):
        result = self.root.step(value)

        initial_pass = result == threshold

        if negation:
            initial_pass = not initial_pass

        return initial_pass

    def print():

        message =  "avaliação tree: \n"

        if self.negation:
            message += "NOT "

        message += self.comparison

        massage += " arvore do nó:\n"

        print(message)
        self.root.print()

        if self.right:
            print("right branch:")
            self.right.print()

        if self.left:
            print("left branch:")
            self.right.print()