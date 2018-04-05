from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

#Setup the tree, so that we have the same type of tree for each itteration, which split on the largest entropy attribute.

dtree = DecisionTreeClassifier(max_depth=1)


def loadData(file):
    #We do not want the firs row, and we do not want the tree to select from first columb.
    df = pd.read_csv(file, sep=',', header=0)
    data = df.values
    return data[...,1:] #Remove index column(we dont need it)

adaboost_train = loadData("dataset/adaboost_train.csv")
#Index 0 is the index, 1 is the y value, rest are attributes for classifying.



def addaBoost(T, trainingData):
    """ Perform boost with decision tree stumps
    Return an matrix of weights assosiated with an classifier h, and the classifier(all trees in our case)"""
    #store the weight assosiated with ech classifier
    weights_classifier = []
    #Seperate attributes from classification
    Y = trainingData[:,0]
    M = len(Y) #number of data entries.
    X = trainingData[...,1:]

    #We want to initialize the weights corresponding to m data.
    sample_w = np.array([1/M]*M) #first element correspond to first data row etc.

    for t in range(T):
        
        pass
        
    tree = dtree.fit(X,Y,sample_weight=sample_w)
    
    prediction = tree.predict(X)
    #We need to take the prediction and figure out how many are wrong, and adjust those
    E_wrong = np.array([int(i) for i in (prediction != Y)]) #create list of which index is wrong
    E_correct = np.array([int(i) for i in (prediction == Y)]) #create list of which index is wrong

    #We know that the sum of this one represent the number e
    e_t = sum(E_wrong)/M #number of mistakes/ number of data entries.

    alpha_t = 1/2 * np.log((1-e_t)/e_t) 
    weights_classifier.append(alpha_t) #store the weight assosiated with this classifier.

    #Need to update the sample_weights for next itteration.

    Z_2 = 0
    for m, weight in enumerate(sample_w):
        #Need to calculate
        Z_2 += weight*np.exp(-alpha_t*prediction[m]*Y[m])
        pass
    print("Z_2 {}".format(Z_2))
    sample_test = np.array(list(sample_w)) #Create copy
    print(sample_test)
    for m in range(M):
        sample_test[m] = sample_test[m]*np.exp(-alpha_t*prediction[m]*Y[m])/Z_2
    print(sum(sample_test))

    #We need to know how much to change the weights.
    Z = sum([w*np.exp(alpha_t*Y[i]*prediction[i]) for i,w in enumerate(sample_w)])
    #print(Z)

    #From wikipedia: Sum_weigths_correct_t+1/Sum_weigths_wrong_t+1 = Sum_weigths_correct_t/Sum_weigths_wrong_t
    #Which simplifies calculating the new weights.
    #Weights of all correct is 1/2, weight of all wrong is 1/2 as well.
    #we can simplify the calculation of new weights.
    Z_c = sum(np.multiply(sample_w,E_correct))
    Z_w = sum(np.multiply(sample_w,E_wrong))

    #difference in the new weights
    W_correct = Z_c/sum(E_correct)
    W_wrong = Z_w/sum(E_wrong)

    Z_c = sum(np.multiply(sample_w,E_correct))
    Z_w = sum(np.multiply(sample_w,E_wrong))
    
    #print(Y)
    #print(tree.tree_.__getstate__()['nodes'])
    for i,wrong in enumerate(E_wrong):
        if(wrong == 1):
            sample_w[i] == sample_w[i]+W_wrong
        else:
            sample_w[i] == sample_w[i]-W_correct
        
    print(sum(sample_w))
    print(sample_w)


addaBoost(100, adaboost_train)

def calculateWeights(M,sample_w, prediction,E_wrong, E_correct):
    """ Update weights """
    Z_c = sum(np.multiply(sample_w,E_correct))
    Z_w = sum(np.multiply(sample_w,E_wrong))

    #difference in the new weights
    W_correct = Z_c/sum(E_correct)
    W_wrong = Z_w/sum(E_wrong)

    Z_c = sum(np.multiply(sample_w,E_correct))
    Z_w = sum(np.multiply(sample_w,E_wrong))

    for i,wrong in enumerate(E_wrong):
        if(wrong == 1):
            sample_w[i] == sample_w[i]+W_wrong
        else:
            sample_w[i] == sample_w[i]-W_correct

    return sample_w

def calculateWeightsTwo(sample_w, Z):
    pass