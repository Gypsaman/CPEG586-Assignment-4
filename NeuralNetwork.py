import sys
import numpy as np
import NNModel as nn
import os
import cv2
from sklearn.utils import shuffle
import random
from ActivationType import ActivationType

def get_data(dir):
    images = os.listdir(dir)
    numimages = len(images)
    X = np.empty((28,28,numimages),dtype='float64')
    Y = np.zeros((10,numimages))

    i = 0
    for image in shuffle(images):
        digit = int(image[0])
        Y[digit,i] = 1.0
        im = cv2.imread("{0}/{1}".format(dir,image),0)
        X[:,:,i] = im/255.0
        i += 1
    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2])
    return X,Y

def runmodel(batchsize,trainX,trainY,testX,testY,epoch,hiddenlayer,batchnorm=False):

    model = nn.Model(x_inputs=784,layers=[(hiddenlayer,ActivationType.SIGMOID),(10,ActivationType.SIGMOID)],number_epochs=epoch,batch_size=batchsize,stochastic=False,lr=0.1)
    model.SetBatchNorm(batchnorm)
    model.fit(trainX,trainY)
    y_predict = model.predict(testX)

    matches = 0
    for i in range(testX.shape[1]):
        index = y_predict[:,i].argmax(axis=0)
        if testY[index,i] == 1:
            matches += 1
    return matches

def resultplot(results):
    import matplotlib.pyplot as plt

    epochs = [25,50,100,150]
    fig = plt.figure()
    sgd = fig.add_subplot(1,2,1)
    sgd.set_title('SGD')
    sgd.set_xlabel("Hidden Layers")
    sgd.set_ylabel("% Predicted")
    for i in range(4):
        sgd.plot([25,50,100,150],results[0,i,:]/100,label='epoch-'+str(epochs[i]))
    sgd.legend()

    bat = fig.add_subplot(1,2,2)
    bat.set_title('Mini Batch')
    for i in range(4):
        bat.plot([25,50,100,150],results[1,i,:]/100,label='epoch-'+str(epochs[i]))
    bat.set_yticklabels([])
    bat.set_xlabel("Hidden Layers")
    bat.legend()
    plt.show()
    plt.savefig('c:\import\image')
def main():
    #resultplot()
    trainX,trainY = get_data('c:/users/cgarcia/CPEG586/minst/Training')
    testX,testY = get_data('c:/users/cgarcia/CPEG586/minst/Test')
    np.random.seed(1105)
    matchesoff = runmodel(20,trainX,trainY,testX,testY,100,100,False)
    matcheson = runmodel(20,trainX,trainY,testX,testY,100,100,True)
    print("BatchNorm on ",matcheson)
    print("BatchNorm off ",matchesoff)
    return
    hiddenlayers = [25,50,100,150]
    epochs = [25,50,100,150]
    results = np.zeros((2,4,4))
    epochnum = 0
    for epoch in epochs:
        hiddenlayernum = 0
        for hiddenlayer in hiddenlayers:
            matches = runmodel(20,trainX,trainY,testX,testY,epoch,hiddenlayer)
            results[0,epochnum,hiddenlayernum] = matches
            matches = runmodel(10,trainX,trainY,testX,testY,epoch,hiddenlayer)
            results[1,epochnum,hiddenlayernum] = matches
            hiddenlayernum += 1
        epochnum += 1
    resultplot(results)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
