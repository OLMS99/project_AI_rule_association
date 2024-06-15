import random
import math
import numpy as np
import time

seed = 1
np.random.seed(seed)

import ActivationFunctions
import LossFunctions
import Node

#antecedent of a rule is the threshold of the node
#consequent of a rule is the children of the node
def co_t_norm(u):
    return min(sum(u), 1)

def t_norm(B,P):
    result = 1
    #for()
    #    result *= co_t_norm()
    return result

def Qcl(ruleSet):
    result = 1
    rmax = len(ruleSet)
    for r in range(rmax):
        result *= max(p(Bj,Pr)) #TODO: implementar p^(Bj|Pr), função de probabilidade de y na classe Bj na premissa Pr
    return result

def Q(ruleSet, Model, beta=0):
    E = None #Todo: minimo erro quadrado em termos de membership function's values of the classification result
    E0 = None #Todo: erro minimo quadrado do modelo inconsequente
    Qac = 1 - (E/E0)
    Qclb = pow(Qcl(rulseSet), beta)
    return Qac * Qclb

def S(A, B):
    AnB = A.intersection(B)
    Asize = len(A)
    Bsize = len(B)
    AnBsize = len(AnB)
    return AnBsize / (Asize + Bsize - AnBsize)

def SRC(i, k, B):
    return S(B[i], B[k])

def SRP(i, k, A):

    result = float('inf')

    for j in len(A):
        term = S(A[i][j], A[k][j])
        result = min(result, term)

    return result

def Cons(Ri, Rk):
    SRPterm = SRP(i, k, R.antecedents)
    SRCterm = SRC(i, k, R.consequents)

    numerador = (SRPterm - SRCterm) * (SRPterm + SRCterm) * SRPterm * SRPterm
    denominador = SRCterm * SRCterm

    return numerador / denominador

def fIncon(R1,R2,i):

    term1 = 0
    for k in range(len(R1)):
        term1 += (1.0 - Con(R1[i],R1[k]))

    term2 = 0
    for l in range(len(R2)):
        term2 += (1.0 - Con(R1[i],R2[l]))

    return term1 + term2

def relevance(A,B,i,x,y):

    result = 0
    #TODO: implementar norma t
    for k in len(x):
        #result += A[i](x[k]) '''norma t''' B[i](y[k])
        continue
    return result

def S(A, B):
    AnB = A.intersection(B)
    Asize = len(A)
    Bsize = len(B)
    AnBsize = len(AnB)
    return AnBsize / (Asize + Bsize - AnBsize)

def SRC(i, k, B):
    return S(B[i], B[k])

def SRP(i, k, A):

    result = float('inf')

    for j in len(A):
        term = S(A[i][j], A[k][j])
        result = min(result, term)

    return result

def Cons(i, R):
    result = 0
    for k in range(len(R)):
        if k == i:
            continue
        result += Cons(i, k, R)
    return result

def Cons(i, k, R):
    SRPterm = SRP(i, k, R.getAntecedent())
    SRCterm = SRC(i, k, R.getConsequent())

    numerador = (SRPterm - SRCterm) * (SRPterm + SRCterm) * SRPterm * SRPterm
    denominador = SRCterm * SRCterm

    return numerador / denominador

def fIncon(R1,R2,i):

    term1 = 0
    for k in range(len(R1)):
        term1 += (1.0 - Con(R1[i],R1[k]))

    term2 = 0
    for l in range(len(R2)):
        term2 += (1.0 - Con(R1[i],R2[l]))

    return term1 + term2

def l1_Lasso(y_true, y_pred, penalidade, weights):

    sum1 = 0
    for i in range(len(y_true)):
        residuo = y_true[i] - y_pred[i]
        sum1 += residuo * residuo

    sum2 = 0
    for i in weights:
        for j in i:
            sum2 += abs(j)

    return sum1 + (penalidade * sum2)

def l2_Ridge(y_true, y_pred, penalidade, weights):
    sum1 = 0
    for i in range(len(y_true)):
        residuo = y_true[i] - y_pred[i]

        sum1 += residuo * residuo

    sum2 = 0
    for i in weights:
        for j in i:
            sum2 += j * j

    return sum1 + (penalidade * sum2)

def elastic_net(y_true, y_pred, penalidade, porcentagem, weights):
    lasso = l1_Lasso(y_true, y_pred, penalidade, weights)
    ridge = l2_Ridge(y_true, y_pred, penalidade, weights)

    return (lasso * (1-porcentagem)) + (ridge * porcentagem)