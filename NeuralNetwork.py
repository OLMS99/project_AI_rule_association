import random
import math
import numpy as np
import time
import copy

import ActivationFunctions as ACT
import LossFunctions as Loss
import ModelMetrics as metrics
import Utils
from collections import Counter
class nnf():
    def __init__(self, layer_sizes = None, act_funcs = None, loss = None, loss_prime = None, seed = 1, params = None, update_weights = True, update_bias = True, debug = False):
        if params is not None:
            return self.load_params(params)

        self.layer_sizes = layer_sizes
        self.act_func = act_funcs
        self.layer_num = len(layer_sizes)
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
        self.accuracy_metric = metrics.Compute_Acc_naive
        self.loss = loss
        self.loss_prime = loss_prime

        self.update_weights = update_weights
        self.update_bias = update_bias

        self.params = self.init_weights(debug)

    def get_params(self):
        return self.params

    def copy(self):
        return nnf(params = copy.deepcopy(self.params))

    def print(self):
        print(vars(self))
        print("dicionario de parametros")
        print(self.params)

    def equal(self, model):
        if not isinstance(model, nnf):
            return False

        if not self.equalParams(model.get_params()):
            return False

        return True

    def equalParams(self, params):
        result = True
        for k,v in self.params.items():
            if k not in params:
                return False
            if isinstance(v, list):
                result = result and Counter(v) == Counter(params[k])

        return result

    def copyParams(self):
        return copy.deepcopy(self.params)

    def load_params(self, params):
        self.params = params

        self.layer_sizes = params["layer sizes"]
        self.act_func = params["ACT functions"]
        self.layer_num = params["num layers"]
        self.input_size = params["input size"]
        self.output_size = params["output size"]
        self.loss = params["loss"]
        self.loss_prime = params["loss prime"]

        self.update_weights = params["update weights"]
        self.update_bias = params["update bias"]

    def getWeights(self):

        returnValues = []
        for i in range(self.layer_num):
            returnValues.append(self.params["W"+str(i+1)])

        return returnValues

    def getAtributes(self):

        returnValues = []
        for i in range(self.layer_num):
            returnValues.append(self.params["A"+str(i)])

        return returnValues

    def set_acc_metric(self, metric):
        self.accuracy_metric = metric

    def init_weights(self, debug=False):

        params = dict()

        params["loss"] = self.loss
        params["loss prime"] = self.loss_prime
        params["ACT functions"] = self.act_func
        params["num layers"] = self.layer_num
        params["layer sizes"] = self.layer_sizes
        params["input size"] = self.input_size
        params["output size"] = self.output_size

        for x in range(self.layer_num-1):
            #params["W"+str(x+1)] = np.nan_to_num(np.random.randn(self.layer_sizes[x+1], self.layer_sizes[x]), nan = 1.0, copy=False)
            #params["b"+str(x+1)] = np.nan_to_num(np.random.randn(self.layer_sizes[x+1], 1), nan = 1.0, copy=False)
            params["W"+str(x+1)] = np.nan_to_num(np.random.rand(self.layer_sizes[x+1], self.layer_sizes[x]), nan = 1.0, copy=False)
            params["b"+str(x+1)] = np.nan_to_num(np.random.rand(self.layer_sizes[x+1], 1), nan = 1.0, copy=False)
            params["f"+str(x+1)] = self.act_func[x]

        if debug:
            for x in range(1,numberLayers):
                print("W%s foi criado(%s): %s" % (x, params["W"+str(x)].shape, params["W"+str(x)]))
                print("b%s foi criado(%s): %s" % (x, params["b"+str(x)].shape, params["b"+str(x)]))
                print("f%s foi criado: %s" % (x, params["f"+str(x)].__name__))

        params["update weights"] = self.update_weights
        params["update bias"] = self.update_bias

        return params

    def forward_pass(self, x_train, debug=False):

        params = self.params

        if debug:
            print("valor de entrada %s: %s" % (x_train.shape, x_train))

        self.params["A0"] = x_train

        for X in range(self.layer_num-1):
            self.params["Z"+str(X+1)] = np.dot(self.params["W"+str(X+1)], self.params["A"+str(X)]) + self.params["b"+str(X+1)]
            self.params["A"+str(X+1)] = self.params["f"+str(X+1)](self.params["Z"+str(X+1)])

            if debug:
                print("valor de Z%d %s: %s = %s dot %s + %s" % (X+1, self.params["Z"+str(X+1)].shape, self.params["Z"+str(X+1)], params["Z"+str(X+1)], self.params["A"+str(X)], params["b"+str(X+1)]))
                print("valor de A%d %s: %s = ACT(%s)" % (X+1, self.params["A"+str(X+1)].shape, self.params["A"+str(X+1)], self.params["Z"+str(X+1)]))

        return self.params["A"+str(self.layer_num-1)]

    def backward_pass(self, y_correct, output, weightUpdate = True, biasUpdate = True, debug = False):

        params = self.params
        change = {}

        diff = self.loss_prime(y_correct, output)
        #diff = y_correct - output

        if debug:
            print("diff %s: %s" % (diff.shape, diff))

        inter_layer_error = diff
        delta = diff * params["f"+str(self.layer_num - 1)](params["Z"+str(self.layer_num - 1)], prime=True)

        if weightUpdate:
            change["W"+str(self.layer_num - 1)] = delta * params["A"+str(self.layer_num - 2)].T

            if debug:
                print("delta de W%d %s: deltaW = %s dot %s" % (self.layer_num - 1, change["W"+str(self.layer_num - 1)].shape, delta.shape, params["Z"+str(self.layer_num - 1)].T.shape))
                print("delta de W%d %s: %s = %s dot %s " % (self.layer_num - 1, change["W"+str(self.layer_num - 1)].shape, change["W"+str(self.layer_num - 1)], delta, params["Z"+str(self.layer_num - 1)].T))

        if biasUpdate:
            change["b"+str(self.layer_num - 1)] = np.sum(delta, axis=0, keepdims=True).reshape(-1,1)

            if debug:
                print("delta de b%d %s: deltab = sum(%s) " % (self.layer_num - 1, change["b"+str(self.layer_num - 1)].shape, delta.shape))
                print("delta de b%d %s: %s = sum(%s) " % (self.layer_num - 1, change["b"+str(self.layer_num - 1)].shape, change["b"+str(self.layer_num - 1)], delta))
        inter_layer_error = delta

        for X in reversed(range(1, self.layer_num-1)):

            act_grad = np.dot(params["W"+str(X+1)].T, inter_layer_error) * params["f"+str(X)](params["Z"+str(X)], prime=True)

            if debug:
                print("delta da camada %d: %s: delta = %s %s_prime(Z-1) X %s" % (X, act_grad.shape, params["Z"+str(X)].shape, params["f"+str(X)].__name__, np.dot(params["W"+str(X+1)].T, inter_layer_error).shape))
                print("gradient de ativação da camada %d: %s: %s = %s X %s_prime(%s) " % (X,act_grad.shape, act_grad, inter_layer_error, params["f"+str(X)].__name__, params["Z"+str(X)]))

            inter_layer_error =  np.dot(params["W"+str(X)].T,act_grad)

            if debug:
                print("error da camada %d %s: %s =  %s dot %s" % (X, inter_layer_error.shape, inter_layer_error, params["W"+str(X)].T, act_grad))

            if weightUpdate:
                change["W"+str(X)] = np.dot(act_grad, params["A"+str(X-1)].T)

                if debug:
                    print("delta de W%d %s: deltaW = %s dot %s" % (X, change["W"+str(X)].shape, act_grad.shape, params["Z"+str(X)].T.shape))
                    print("delta de W%d %s: %s = %s dot %s " % (X, change["W"+str(X)].shape, change["W"+str(X)], act_grad, params["A"+str(X-1)].T))

            if biasUpdate:
                change["b"+str(X)] = np.sum(act_grad, axis=0, keepdims=True).reshape(-1,1)

                if debug:
                    print("delta de b%d %s: deltab = sum(%s) " % (X, change["b"+str(X)].shape, act_grad.shape))
                    print("delta de b%d %s: %s = sum(%s) " % (X, change["b"+str(X)].shape, change["b"+str(X)], act_grad))

        return change

    def update_params(self, changes, debug=False):

        for key, value in changes.items():
            temp1 = self.params[key]
            temp2 = self.learning_rate * value

            if debug:
                print("======%s - %s======" % (temp1.shape, temp2.shape))
                print("%s - %s * %s = " % (temp1, value, self.learning_rate))

            self.params[key] -= temp2

            if debug:
                print("%s: %s" % (self.params[key], self.params[key].shape))

    def prune_input(self, Neurons):

        target = self.get_params()

        if "W1" not in self.params:
            return

        weight_holder = target["W1"]
        for idx in Neurons:
            weight_holder[:, idx] = np.zeros(weight_holder.shape[0])
        target["W1"] = weight_holder

        return nnf(params = target)

    def train(self, X_train, y_train, X_val, y_val, epochs=25, learning_rate=0.01, update_weights = True, update_bias = True, debug=False):
        self.epochs = epochs
        self.learning_rate = learning_rate
        Debug = debug

        start_time = time.time()

        indexes_original = []
        for i in range(len(X_train)):
            indexes_original.append(i)

        finalAcc = 0.0
        for iteration in range(self.epochs):
            indexes = indexes_original 
            np.random.shuffle(indexes)
            for j in indexes:
                if debug:
                    print("============== X = %s y = %s ==================" % (X_train[j], y_train[j]))

                input_ = X_train[j].reshape(-1,1)
                correct_output_ = y_train[j].reshape(-1,1)
                output = self.forward_pass(input_, debug = Debug).reshape(-1,1)
                changes = self.backward_pass(correct_output_, output, weightUpdate = update_weights, biasUpdate = update_bias, debug = Debug)
                self.update_params(changes)

            if debug:
                print("shape do dado de entrada: (%s, %s)" % (X_val.shape[0],X_val.shape[1]))
                print("shape do dado de saida: (%s, %s)" % (y_val.shape[0],y_val.shape[1]))
                print("tamanho da saida da rede neural: %s" % (self.output_size))

            y_pred = np.zeros(shape=(X_val.shape[0], self.output_size))
            for i, sample in enumerate(X_val):
                y_pred[i] = self.predict(sample)

            accuracy = metrics.Compute_Acc_naive(y_pred, y_val)

            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(
                iteration+1, time.time() - start_time, accuracy
            ))
            finalAcc = accuracy

        return finalAcc

    def cost(self, prediction, true_y):
        m = true_y.shape[1]
        term = np.dot(true_y,np.log(prediction))
        term_complement = np.dot((1-true_y),np.log(1-prediction))
        sum_cost = np.sum(term + term_complement)
        return -sum_cost/m

    def predict(self, X):

        output = self.forward_pass(X.reshape(-1,1))
        return np.squeeze(output, axis=None)

    def getInputHiddenLayers(self, InputLayer = False):
        results = []

        if InputLayer:
            for i in range(self.layer_num - 1):
                results.append(self.params["A"+str(i)])

            return results

        for i in range(1, self.layer_num - 1):
            results.append(self.params["A"+str(i)])

        return results

    def accuracy(self, X, y):
        y_pred = []
        for idx, x_pred in enumerate(X):
            y_pred.append(self.predict(X[idx, :]))
        y_pred = np.array([np.array(yi) for yi in y_pred])

        return self.accuracy_metric(y_pred, y)

    def destroy(self):
        self.params.clear()