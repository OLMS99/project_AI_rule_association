import NeuralNetwork as NN
import Node
import NodeMofN
import ActivationFunctions as ACT
import LossFunctions as Loss
import TreeHeader

#teste da função cópia de um nó
def testCopiaNo():
    noTest = Node.Node(featureIndex=0, layerIndex=1, threshold = 2, comparison = ">=", value="StringValida")
    print(vars(noTest))
    noTest.print()
    print("#####################################")
    copiaTest = noTest.copy_node()
    copiaTest.comparison = "!="
    copiaTest.value = "ValorInvalido"
    print(vars(noTest))
    noTest.print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(vars(copiaTest))
    copiaTest.print()
    print("======================================")

#teste da função cópia de um nó MofN
def testCopiaNoMofN():
    MofNTest = NodeMofN.NodeMofN(featureIndex=11, layerIndex=22, threshold=33, listaPremissas=[['A','a'],['B','b'],['C','c']], negation=True)
    print(vars(MofNTest))
    MofNTest.print()
    print("#####################################")
    copiaTest = MofNTest.copy_node()
    copiaTest.featureIndex=0
    copiaTest.threashold=-1
    copiaTest.set_comparisons.remove(['A','a'])
    copiaTest.set_comparisons.remove(['C','c'])
    print(vars(MofNTest))
    MofNTest.print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(vars(copiaTest))
    copiaTest.print()
    print("======================================")

#teste da função cópia de uma arvore
def testCopiaArvore():
    noTest1 = Node.Node(featureIndex=3, layerIndex=4, threshold = 5, comparison = ">=")
    noTest2 = Node.Node(featureIndex=6, layerIndex=7, threshold = 8, comparison = "<")
    noTest3 = Node.Node(featureIndex=10, layerIndex=15, threshold = 20, comparison = "==", value="folha")
    noTest4 = Node.Node(featureIndex=77, layerIndex=88, threshold = 99, comparison = "!=", value="raiz")

    noTest1.set_right(noTest3)
    noTest1.set_left(noTest2)
    noTest2.set_right(noTest4)
    noTest1.print()
    print("#####################################")
    copiaTest = noTest1.copy_tree()
    noTest1.set_right(None)
    noTest1.print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    copiaTest.print()
    print("======================================")

def testCopiaHeader():
    noTest1 = Node.Node(featureIndex=3, layerIndex=4, threshold = 5, comparison = ">=")
    noTest2 = Node.Node(featureIndex=6, layerIndex=7, threshold = 8, comparison = "<")
    noTest3 = Node.Node(featureIndex=10, layerIndex=15, threshold = 20, comparison = "==", value="folha")
    noTest4 = Node.Node(featureIndex=77, layerIndex=88, threshold = 99, comparison = "!=", value="raiz")

    noTest1.set_right(noTest3)
    noTest1.set_left(noTest2)
    noTest2.set_right(noTest4)
    treeHead = TreeHeader.TreeHeader(tree = noTest1.copy_tree())
    treeHead.print()
    print("#####################################")
    copiaHead = treeHead.copy()
    copiaHead.set_left(copiaHead.tree, None)
    copiaHead.print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    treeHead.print()
    print("======================================")

#teste da função cópia de uma rede neural
def testCopiaRede():
    redeTeste = NN.nnf(layer_sizes=[4,5,2], act_funcs=[ACT.sigmoid,ACT.sigmoid,ACT.sigmoid], loss=Loss.mae, loss_prime=Loss.mae_prime)
    redeTeste.print()
    print("#####################################")
    copiaTest = redeTeste.copy()
    copiaTest.params["update bias"] =False
    copiaTest.params["update weights"] =False
    copiaTest.params["num layers"] = 99
    copiaTest.params["ACT functions"][1] = ACT.ReLU
    copiaTest.params["layer sizes"][1] = 3
    copiaTest.params["W1"][0][2] = float('inf')
    copiaTest.params["W2"][0][1] = float('inf')
    copiaTest.params["b1"][3] = float('inf')
    copiaTest.params["b2"][0] = float('inf')
    redeTeste.print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    copiaTest.print()
    print("======================================")
    print(copiaTest.params["update bias"]==redeTeste.params["update bias"])
    print(copiaTest.params["update weights"]==redeTeste.params["update weights"])
    print(copiaTest.params["num layers"]==redeTeste.params["num layers"])
    print(copiaTest.params["ACT functions"][1]==redeTeste.params["ACT functions"][1])
    print(copiaTest.params["layer sizes"][1]==redeTeste.params["layer sizes"][1])
    print(copiaTest.params["W1"][0][2]==redeTeste.params["W1"][0][2])
    print(copiaTest.params["W2"][0][1]==redeTeste.params["W2"][0][1])
    print(copiaTest.params["b1"][3]==redeTeste.params["b1"][3])
    print(copiaTest.params["b2"][0]==redeTeste.params["b2"][0])
    print(redeTeste.equal(copiaTest))

#testCopiaNo()
#testCopiaNoMofN()
#testCopiaArvore()
#testCopiaRede()
testCopiaHeader()