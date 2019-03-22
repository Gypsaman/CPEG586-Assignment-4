import numpy as np
import math
from ActivationType import ActivationType
class LROptimizerType(object):
    NONE = 1
    ADAM = 2
def loss(predict,y,AF):
    if AF == ActivationType.SOFTMAX:
        Loss = (y*np.log(predict+0.001)).sum()
    else:
        Loss = (0.5 * (np.multiply((predict-y),(predict-y)))).sum()/predict.shape[1]
    return Loss

def relu(data):
    sh = data.shape
    result = np.array([[x if x > 0 else 0 for x in row] for row in data]).reshape(sh)
    return result

def bprelu(dA,activation):
    return dA

def sigmoid(data):
    result = 1/(1+np.exp(-1*data))
    return result

def bpsigmoid(dA,activation):
    return dA*(activation*(1-activation))

def tanh(data):
    result = (np.exp(data)-np.exp(-1*data)/(np.exp(data)+np.exp-1*(data)))
    return result

def bptanh(dA,activation):
    return dA*(1-activation*activation)

def softmax(data):
    esum = np.exp(data).sum()
    result = np.exp(data)/esum
    return 'a'

def bpsoftmax(dA,activation):
    return dA

Activations = [sigmoid,tanh,relu,softmax]
Gradients = [bpsigmoid,bptanh,bprelu,bpsoftmax]

class Layer(object):
    """description of class"""
    numparams = 2
    epsilon = 1.0e-9
    AdamBeta1 = 0.9
    AdamBeta2 = 0.999
    
    def __init__(self, neurons, inputs, activationf=ActivationType.RELU, BatchNorm = False):

        self.layerNodes = neurons
        self.inputNodes = inputs

        self.weights = np.random.uniform(low=-0.1,high=0.1,size=(neurons,inputs))
        self.biases = np.random.uniform(low=-1,high=1,size=(neurons,1))
        self.activationf = activationf
        self.prev_a = np.zeros((inputs,1))
        self.delta = np.zeros((neurons,1))
        self.dw = np.zeros(neurons)
        self.db = np.zeros(neurons)
        self.activation = np.zeros((neurons,1))

        # batchNorm
        self.isBatchNorm = BatchNorm
        self.BNBeta = np.random.rand(1)
        self.BNgamma = np.random.rand(1)
        self.Shat = np.zeros((neurons,inputs))
        self.Sb = np.zeros((neurons,inputs))
        self.mu = np.zeros((neurons))
        self.sigma2 = np.zeros((neurons))
        self.gamma = np.random.rand(1)
        self.beta = np.random.rand(1)
        self.runningmu = np.zeros((neurons,1))
        self.runningsigma2 = np.zeros((neurons,1))

        #Adam Variables
        self.mt = np.zeros((Layer.numparams,neurons))
        self.vt = np.zeros((Layer.numparams,neurons))

    def forward(self,prev_activation,isTraining = True):
        assert prev_activation.shape[0] == self.inputNodes
        self.prev_a = prev_activation.astype(float)
        s = np.dot(self.weights,self.prev_a) + self.biases
        if self.isBatchNorm == True:
            if(isTraining):
                self.mu = np.mean(s,axis=1).reshape((self.layerNodes,1))
                self.sigma2 = np.var(s,axis=1).reshape((self.layerNodes,1))
                self.runningmu = 0.9 * self.runningmu + (1-0.9) * self.mu
                self.runningsigma2 = 0.9 * self.runningsigma2 + (1-0.9) * self.sigma2
            else:
                self.mu = self.runningmu
                self.sigma2 = self.runningsigma2

            self.Shat = (s - self.mu)/np.sqrt(self.sigma2 + Layer.epsilon)
            self.Sb = self.Shat * self.gamma + self.beta
            s = self.Sb

        self.activation = Activations[self.activationf](s)
        
        return self.activation

    def backprop(self,dA):
        m = self.prev_a.shape[1]
        
        self.delta = Gradients[self.activationf](dA,self.activation)
        
        if self.isBatchNorm == True:
            self.dbeta = np.sum(self.delta,axis=1).reshape((self.layerNodes,1))
            self.dgamma = np.sum(self.delta*self.Shat,axis=1).reshape((self.layerNodes,1))
            self.deltabn = (self.delta * self.gamma)/(m*np.sqrt(self.sigma2+Layer.epsilon)) * (m-1-(self.Shat*self.Shat))

            self.dw = 1/m*np.dot(self.deltabn,self.prev_a.T)
            self.db = 1/m*np.sum(self.deltabn,axis=1,keepdims=True)

        else:
            self.dw = 1/m*np.dot(self.delta,self.prev_a.T)
            self.db = 1/m*np.sum(self.delta,axis=1,keepdims=True)
        da_prev = np.dot(self.weights.T,self.delta)

        return da_prev

    def Adam(self,t):

        #Adam calc for W
        self.mt[0] = Layer.AdamBeta1*self.mt[0] + (1-Layer.AdamBeta1)*self.dw
        self.vt[0] = Layer.AdamBeta2*self.vt[0] + (1-Layer.AdamBet2)*self.dw*self.dw
        mtBiased = self.mt[0]/(1-np.power(Layer.AdamBeta1,t))
        vtBiased = self.vt[0]/(1-np.power(Layer.AdamBeta2,t))
        self.dw = mtBiased *(1/(np.sqrt(vtBiased)-epsilon))
        #Adam calc for b
        self.mt[1] = Layer.AdamBeta1*self.mt[1] + (1-Layer.AdamBeta1)*self.db
        self.vt[1] = Layer.AdamBeta2*self.vt[1] + (1-Layer.AdamBeta2)*self.db*self.db
        mtBiased = self.mt[1]/(1-np.power(Layer.AdamBeta1,t))
        vtBiased = self.vt[1]/(1-np.power(Layer.AdamBeta2,t))
        self.db = mtBiased *(1/(np.sqrt(vtBiased)-Layer.epsilon))                               

        da_prev = np.dot(self.weights.T,self.delta)
        return da_prev

    def update_parameters(self,lr,optimizer = LROptimizerType.NONE,iternum=0):
        if optimizer == LROptimizerType.NONE:
            self.weights -= lr * self.dw
            self.biases -=  lr * self.db
        elif optimizer == LROptimizerType.ADAM:
            Adam(iternum)
        if self.isBatchNorm == True:
            self.beta = self.beta - lr * self.dbeta
            self.gamma = self.gamma - lr * self.dgamma


class Model(object):
    """description of class"""
    def __init__(self, x_inputs, layers, number_epochs=10, batch_size=1, stochastic=False,lr=0.1):
        self.layers = []
        self.epochs =  number_epochs
        self.batchsize = batch_size
        self.stochastic = stochastic
        self.lr = lr
        self.lastlayerAF = layers[len(layers)-1][1]
        self.optimizer = LROptimizerType.NONE

        prev_a = x_inputs
        for size,actf in layers:
            layer = Layer(neurons=size,inputs=prev_a,activationf=actf)
            self.layers.append(layer)
            prev_a = size
    
    def SetBatchNorm(self,Turnon):
        for layer in self.layers[:-1]:
            layer.isBatchNorm = Turnon
    def fit(self,X,Y):
        self.samples = X.shape[0]
        batches = math.floor(self.samples / self.batchsize)
        if batches * self.batchsize < self.samples:
            batches += 1

        for epoch in range(self.epochs):
            cost = 0
            iteration = 0
            for batch in range(batches):
                startbatch = batch * self.batchsize
                endbatch = min((batch+1)*self.batchsize,self.samples)
                a_prev = X[:,startbatch:endbatch ]
                y_sample = Y[:,startbatch: endbatch]
                ## Forward Calculate
                for layer in self.layers:
                    a_prev = layer.forward(a_prev,isTraining=True)
                cost += loss(a_prev,y_sample,self.lastlayerAF)  
                # Backward Propagation
                da_prev = -(y_sample-a_prev)  
                for layer in reversed(self.layers):
                    da_prev = layer.backprop(da_prev).astype(float)
                    layer.update_parameters(self.lr,self.optimizer,iteration)
                iteration += 1
            print("epoch =" + str(epoch) + " loss = " + str(cost)) 
    
    def predict(self,X):
        a_prev = X
        for layer in self.layers:
            a_prev = layer.forward(a_prev,isTraining=False)
        return a_prev






