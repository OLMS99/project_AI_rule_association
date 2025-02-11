from Node import Node
from collections import deque

class TreeHeader:
    def __init__(self, tree = None):
        self.antecedents = dict()
        self.consequents = dict()
        if tree is not None:
            self.tree = tree
            self.extract()

    def set_tree(self, arvore):
        if not isinstance(arvore, node) or arvore is None:
            return
        self.tree = arvore
        self.extract()

    def get_antecedents(self):
        return self.antecedents.copy()

    def get_consequents(self):
        return self.consequents.copy()

    def print(self):
        self.tree.print()
        print("Number of antecedents: ", len(self.antecedents))
        print("Number of consequents: ", len(self.consequents))

    def extract(self):
        self.antecedents.clear()
        self.consequents.clear()
        queue = deque([self.tree])
        first=True
        while queue:
            noNovo = queue.popleft()

            if first:
                first=False
                if noNovo.is_leaf_node():
                    self.consequents[noNovo.__hash__()] = (None, 0, noNovo.get_node_info())
                else:
                    self.antecedents[noNovo.__hash__()] = (None, 0, noNovo.get_node_info())

            if noNovo.left:
                if noNovo.left.is_leaf_node():
                    self.consequents[noNovo.left.__hash__()] = (noNovo, -1, noNovo.left.get_node_info())
                else:
                    self.antecedents[noNovo.left.__hash__()] = (noNovo, -1, noNovo.left.get_node_info())
                    queue.append(noNovo.left)

            if noNovo.right:
                if noNovo.right.is_leaf_node():
                    self.consequents[noNovo.right.__hash__()] = (noNovo, 1, noNovo.right.get_node_info())
                else:
                    self.antecedents[noNovo.right.__hash__()] = (noNovo, 1, noNovo.right.get_node_info())
                    queue.append(noNovo.right)

    def set_right(self, noPai, novoNo):
        print("iniciado anexação a direita")
        if self.antecedents.get(noPai.__hash__()) is None:
            return

        noRemovido = noPai.set_right(novoNo)

        if noRemovido is not None:
            self.removeNode(noRemovido)

        if novoNo is None:
            return

        if novoNo.is_leaf_node():
            self.consequents[novoNo.__hash__()] = (noPai, 1, novoNo.get_node_info())
        else:
            self.antecedents[novoNo.__hash__()] = (noPai, 1, novoNo.get_node_info())

        print("Number of antecedents: ", len(self.antecedents))
        print("Number of consequents: ", len(self.consequents))

    def set_left(self, noPai, novoNo):
        if self.antecedents.get(noPai.__hash__()) is None:
            return

        noRemovido = noPai.set_left(novoNo)

        if noRemovido is not None:
            self.removeNode(noRemovido)

        if novoNo is None:
            return

        if novoNo.is_leaf_node():
            self.consequents[novoNo.__hash__()] = (noPai, -1, novoNo.get_node_info())
        else:
            self.antecedents[novoNo.__hash__()] = (noPai, -1, novoNo.get_node_info())
            self.addNode(noNovo)

    def addNode(self, node, noPai, side):
        if self.antecedents.get(noPai.__hash__()) is None:
            return
        if self.antecedents.get(node.__hash__()) is not None or self.consequents.get(node.__hash__()) is not None:
            return

        queue = deque([node])
        first=True

        while queue:
            noNovo = queue.popleft()

            if first:
                if noNovo.is_leaf_node():
                    self.consequents[noNovo.__hash__()] = (noPai, side, noNovo.get_node_info())
                else:
                    self.antecedents[noNovo.__hash__()] = (noPai, side, noNovo.get_node_info())
                first=False

            if noNovo.left:
                if noNovo.left.is_leaf_node():
                    self.consequents[noNovo.left.__hash__()] = (noNovo, -1, noNovo.left.get_node_info())
                else:
                    self.antecedents[noNovo.left.__hash__()] = (noNovo, -1, noNovo.left.get_node_info())
                queue.append(noNovo.left)

            if noNovo.right:
                if noNovo.right.is_leaf_node():
                    self.consequents[noNovo.right.__hash__()] = (noNovo, 1, noNovo.right.get_node_info())
                else:
                    self.antecedents[noNovo.right.__hash__()] = (noNovo, 1, noNovo.right.get_node_info())
                queue.append(noNovo.right)

    def removeNode(self, node):
        if self.antecedents.get(node.__hash__()) is None and self.consequents.get(node.__hash__()) is None:
            return
        queue = deque([node])

        while queue:
            noRemover = queue.popleft()

            if noRemover.left:
                queue.append(noRemover.left)

            if noRemover.right:
                queue.append(noRemover.right)

            if noRemover.is_leaf_node():
                self.consequents.pop(noRemover.__hash__())
            else:
                self.antecedents.pop(noRemover.__hash__())

    #def filter(self, node):

    def copy(self):
        return TreeHeader(self.tree.copy_tree())

    def copy_tree(self):
        return self.tree.copy_tree()

    def copy_tree_n_node(self,node):
        if self.antecedents.get(node) is None and self.consequents.get(node) is None:
            return

        return self.tree.copy_tree_n_node()

    def destroy(self):
        self.antecedents.clear()
        self.consequent.clear()
        self.tree.destroy()
        del self