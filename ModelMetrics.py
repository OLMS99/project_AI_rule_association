import random
import math
import numpy as np
import time

seed = 1
np.random.seed(seed)

def Where_n(array, n=1):
    positions = []
    for pos,it in enumerate(array):
        if it==n:
            positions.append(pos)
    return positions

def Calculate_TruePositive(confusion_matrix, y):
    if confusion_matrix.shape[0] == 2:
        return confusion_matrix[1][1]
        
    y_pos = Where_n(y)
        
    TruePositive = 0
    for ite in y_pos:
        TruePositive += confusion_matrix[ite][ite]
        
    return TruePositive
    
def Calculate_TrueNegative(confusion_matrix, y):
    if confusion_matrix.shape[0] == 2:
        return confusion_matrix[0][0]
        
    y_pos = Where_n(y, 0)
        
    TrueNegative = 0
    for val in y_pos:
        for pred in y_pos:
            TrueNegative += confusion_matrix[val][pred]
        
    return TrueNegative
    
def Calculate_FalsePositive(confusion_matrix, y):
    if confusion_matrix.shape[0] == 2:
        return confusion_matrix[0][1]
        
    y_val = Where_n(y, 0)
    y_pred = Where_n(y, 1)
        
    FalsePositive = 0
    for val in y_val:
        for pred in y_pred:
            FalsePositive += confusion_matrix[val][pred]
        
    return FalsePositive
    
def Calculate_FalseNegative(confusion_matrix, y):
    if confusion_matrix.shape[0] == 2:
        return confusion_matrix[1][0]
        
    y_val = Where_n(y, 1)
    y_pred = Where_n(y, 0)
        
    FalseNegative = 0
    for val in y_val:
        for pred in y_pred:
            FalseNegative += confusion_matrix[val][pred]
        
    return FalseNegative
    
def Calculate_accuracy(confusion_matrix):
    trueResult = 0
    sizeVal = confusion_matrix.shape[0]
    total = 0
        
    for ite in range(sizeVal):
        trueResult += confusion_matrix[ite][ite]
            
    for val in range(sizeVal):
        for pred in range(sizeVal):
            total += confusion_matrix[val][pred]
            
    return trueResult / total
    
def Calculate_precision(confusion_matrix, trueY):
    truePositive = Calculate_TruePositive(confusion_matrix, trueY)
    falsePositive = Calculate_FalsePositive(confusion_matrix, trueY)
        
    return truePositive/(truePositive + falsePositive)
    
def Calculate_recall(confusion_matrix, trueY):
    truePositive = Calculate_TruePositive(confusion_matrix, trueY)
    falseNegative = Calculate_FalseNegative(confusion_matrix, trueY)
        
    return truePositive/(truePositive + falseNegative)
    
def Calculate_specificity(confusion_matrix, trueY):
    trueNegative = Calculate_TrueNegative(confusion_matrix, trueY)
    falsePositive = Calculate_FalsePositive(confusion_matrix, trueY)
        
    return trueNegative/(trueNegative + falsePositive)
        
def Calculate_fbscore(confusion_matrix, trueY, beta):
    recall = Calculate_recall(confusion_matrix, trueY)
    precision = Calculate_precision(confusion_matrix, trueY)
        
    beta2 = beta * beta
    term1 = beta2 + 1
    denominador = beta2 * precision + recall
    numerador = precision * recall
        
    return term1 * numerador / denominador

def Calculate_macrofb(confusion_matrix, y_val, beta):
    beta2 = beta * beta
    n = y_val.shape[0]
    term1 = (1 + beta2)/n
        
    sum_part = 0
    for y in y_val:
        precision_i = Calculate_precision(confusion_matrix, y)
        recall_i = Calculate_recall(confusion_matrix, y)
        sum_part += precision_i * recall_i/(beta2 * precision_i + recall_i)
        
    return term1 * sum_part
    
def Calculate_microfb(confusion_matrix, y_val, beta):
    beta2 = beta * beta
    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
        
    for y in Y_val:
        sum_TP += Calculate_TruePositive(confusion_matrix, y)
        sum_FP += Calculate_FalsePositive(confusion_matrix, y)
        sum_FN += Calculate_FalseNegative(confusion_matrix, y)
        
    precision = sum_TP /(sum_TP + sum_FP)
    recall = sum_TP /(sum_TP+ sum_FN)
        
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)

def Cohen_Kappa(confusion_matrix, Y_val):
    classes = confusion_matrix.shape[0]
    po = Calculate_accuracy(confusion_matrix)
    pe = 0
    
    N2 = classes * classes
    
    for k in range(classes):
        sum_ak = 0
        sum_bk = 0
        
        for case in range(classes):
            sum_ak += confusion_matrix[k][case]
            sum_bk += confusion_matrix[case][k]
            
        pe += sum_ak * sum_bk
    
    pe = pe / N2
    numerador = po - pe
    denominador = 1 - pe
    return numerador/denominador

def variance(y):
    mean = np.mean(y)
    result = 0
    for yi in y:
        diff = yi - mean
        result += diff * diff
        
    return result

def Rsquared(y_pred, y_val):
    numerador = mse(y_val, y_pred)
    denominador = variance(y_val)
    
    return 1 - (numerador/ denominador)
 
def RsquaredAdjusted(y_pred, y_val):
    R2 = Rsquared(y_pred, y_val)
    n = y_val.shape[0]
    k = np.mean(y_val)
    
    denominador = n - k - 1
    numerador = (R2 - 1) * (n - 1)
    return 1 - (numerador/denominador)
    
def Compute_AccR(y_pred, y_val):
    erros = 0
    output_length = y_val.shape[0]
    
    for indice, y in enumerate(y_val):
        pred = np.round(y_pred[indice])
        
        if output_length == 1:
            if (pred != y).all():
                erros += 1
                
        else:
            for i in range(output_length):
                if pred[i] != y[i]:
                    erros +=1
                    
    porcentagemErro = erros/(y_val.shape[0] * y_val.shape[1])
    return 1 - porcentagemErro
    
def Compute_Acc_naive(y_pred, y_val):
    acertos = 0
    num_cases = y_val.shape[0]
    output_length = y_val.shape[1]
        
    for indice, y in enumerate(y_val):
        pred = np.round(y_pred[indice])
        
        if output_length == 1:
            if (pred == y).all():
                acertos += 1
        
        else:
            for i in range(output_length):
                if pred[i] == y[i]:
                    acertos +=1
            
                    
    return acertos/(num_cases * output_length)
    
def Compute_GM(matrix_confusao, y):
    acc_positive = Calculate_TruePositive(matrix_confusao,y)
    acc_negative = Calculate_TrueNegative(matrix_confusao,y)
        
    return math.sqrt(acc_positive * acc_negative)
    
def Compute_matrix(y_pred, y_val):
        
    size_results = y_val.shape[0]
            
    if size_results == 1:
        matriz_confusao = np.zeros((2, 2))
            
        for pred, y in zip(y_pred, y_val):
            
            val_pred = int(np.round(pred[0]))
            val_y = int(y[0])
                
            matriz_confusao[val_y][val_pred] += 1
                
        return matriz_confusao
        
    matriz_confusao = np.zeros((size_results, size_results))
        
    for pred, y in zip(y_pred, y_val):
            
        y_pos = Where_n(y)            
        pred_pos = Where_n(np.round(pred))
            
        for pos in y_pos:
            for it in pred_pos:
                matriz_confusao[pos][it]+= 1
                                
    return matriz_confusao