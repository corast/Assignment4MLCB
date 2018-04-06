import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

#Setup the tree, so that we have the same type of tree for each itteration, which split on the largest entropy attribute.

dtree = DecisionTreeClassifier(max_depth=1)


def loadData(file):
    #We do not want the firs row, and we do not want the tree to select from first columb.
    df = pd.read_csv(file, sep=',', header=0)
    data = df.values
    return data[...,1:] #Remove index column(we dont need it)

adaboost_train = loadData("dataset/adaboost_train.csv")
#Index 0 is the index, 1 is the y value, rest are attributes for classifying.


def INCORRECT_calculateWeightsOLD(M,sample_w, prediction,E_wrong, E_correct):
    #From wikipedia: Sum_weigths_correct_t+1/Sum_weigths_wrong_t+1 = Sum_weigths_correct_t/Sum_weigths_wrong_t
    #Which simplifies calculating the new weights.
    #Weights of all correct is 1/2, weight of all wrong is 1/2 
    #we can simplify the calculation of new weights.
    Z_c = sum(np.multiply(sample_w,E_correct))
    Z_w = sum(np.multiply(sample_w,E_wrong))

    #difference in the new weights
    W_correct = Z_c/sum(E_correct)
    W_wrong = Z_w/sum(E_wrong)

    for i,wrong in enumerate(E_wrong):
        if(wrong == 1):
            sample_w[i] = sample_w[i]+W_wrong
        else:
            sample_w[i] = sample_w[i]-W_correct
    #print(sum(sample_w))
    return sample_w

def calculateWeights(M,sample_w, alpha_t,predictions,Y):
    """ Calculate weights """
    Z = 0
    for m, weight in enumerate(sample_w):
        #Need to calculate
        Z += weight*np.exp(-alpha_t*predictions[m]*Y[m])
    
    #for m in range(M):
    #    sample_w[m] = (sample_w[m]*np.exp(-alpha_t*predictions[m]*Y[m]))/Z
    #print(sum(sample_w))
    A = np.multiply(Y,-alpha_t)
    B = np.multiply(predictions,A)
    C = np.multiply(B,Z)
    D = np.exp(C)
    sample_w = np.multiply(sample_w, D)
    #sample_w = np.multiply(sample_w,np.exp(-alpha_t*predictions*Y),Z)
    return sample_w


def addaBoost(T, trainingData, testData):
    """ Perform boost with decision tree stumps
    Return an matrix of weights assosiated with an classifier h, and the classifier(all trees in our case)"""

    #To test tree later on.
    Y_test = testData[:,0]
    X_test = testData[...,1:]
    M_test = len(Y_test)
    #store the weight assosiated with ech classifier
    weights_classifier = []
    classification_predictions = [] #Store the prediction of every tree.
    #Seperate attributes from classification
    Y = trainingData[:,0]
    M = len(Y) #number of data entries.
    X = trainingData[...,1:]

    #We want to initialize the weights corresponding to m data.
    sample_w = np.array([1/M]*M) #first element correspond to first data row etc.
    #print(sample_w.shape)
    for t in range(T):
        tree = dtree.fit(X,Y,sample_weight=sample_w)
    
        predictions = tree.predict(X)
        #predictions2 = tree.predict(X_test)
        #We need to take the prediction and figure out how many are wrong, and adjust those
        E_wrong = np.array([int(i) for i in (predictions != Y)]) #create list of which index is wrong
        #E_correct = np.array([int(i) for i in (predictions == Y)]) #create list of which index is correct(if needed)

        #Now we need to dot with the corresponding weight for that row in the dataset which is wrong.
        e_t = np.dot(sample_w,E_wrong) 

        alpha_t = 1/2 * np.log((1-e_t)/e_t) #How much we need to adjust the new weights.
        weights_classifier.append([alpha_t]) #store the weight assosiated with this classifier, for later use.

        sample_w = calculateWeights(M,sample_w,alpha_t,predictions,Y)
        classification_predictions.append(tree.predict(X_test))
        #We test the tree in the current data.
        #store ever prediction of every classifier.

        predictionMatrix = []

        for prediction in classification_predictions:
            #Check the result thus far for every iteration.
            predictionMatrix.append(list(prediction))
        
        predictionMatrix = np.asarray(predictionMatrix)
        weightMatrix = np.asarray(weights_classifier)
        #print("new line")
        #calculate the sum of a_t*f_t(x)
        guess = np.multiply(weightMatrix, predictionMatrix)
        guess_T = np.transpose(guess)

        #figure out the final verdict from the trees.
        sign_predictions = np.sign(guess_T.sum(axis=1))
        #print(sign_predictions)
        #print(Y_test)
        Error = np.sum(sign_predictions != Y_test)
        #Correct = np.sum(sign_predictions == Y_test)
        #print(Error+Correct)
        #print(correct)
        Error_rate = Error/M_test * 100
        pyplot.plot(t,Error_rate,'r+')
        print("Error rate itteration {} is {}% ".format(t, Error_rate))
        #Calculate the error from this.
    pyplot.ylim([0,50])
    pyplot.xlabel("Itterations")
    pyplot.ylabel("Error rate in %")
    pyplot.show()
    
    return weights_classifier

adaboost_test = loadData("dataset/adaboost_test.csv")
weights = addaBoost(200, adaboost_train, adaboost_test)
#Run som tests



#print(classifiers)