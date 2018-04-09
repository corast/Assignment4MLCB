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

def calculateWeights(M,sample_w, alpha_t,predictions,Y):
    """ Calculate weights """
    #for m, weight in enumerate(sample_w):
        #Need to calculate
        #Z += weight*np.exp(-alpha_t*predictions[m]*Y[m])
    #sample_w_t = np.array(sample_w,copy=True) #Debugging if same result
    #for m in range(M):
    #    sample_w_t[m] = (sample_w[m]*np.exp(-alpha_t*predictions[m]*Y[m]))/Z
    #print(sample_w_t)
    #Re implemented the loops as matrix multiplication, to speed things up.
    #The only downfall, is that once memory is full for each itteration it seems to slow down alot
    A = np.multiply(Y,predictions)
    B = np.multiply(-alpha_t, A)
    C = np.exp(B)
    D = np.multiply(C,sample_w)
    Z = sum(D)
    sample_w = np.multiply(D,1/Z)

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

        #update weights.
        sample_w = calculateWeights(M,sample_w,alpha_t,predictions,Y)
        #Store the prediction of each new classifier. (doesnt matter since we don't need to test on any other dataset)
        classification_predictions.append(tree.predict(X_test))

        #We test the prediction of every tree for ever itteration.

        #get workable narray
        predictionMatrix = np.asarray(classification_predictions)
        #get workable narray
        weightMatrix = np.asarray(weights_classifier)
        #calculate the sum of a_t*f_t(x)
        adaboost_test = np.multiply(weightMatrix, predictionMatrix)
   
        #transpose since we want ever a tree predicion for every column and one prediction on ever row.
        adaboost_test_T = np.transpose(adaboost_test)

        #figure out the final verdict from the trees into one vector for every row of data.
        sign_predictions = np.sign(adaboost_test_T.sum(axis=1))

        Error = np.sum(sign_predictions != Y_test) #add up ever row that is not equal.
        #Correct = np.sum(sign_predictions == Y_test)

        #Calculate the error from this.
        Error_rate = Error/M_test * 100
        pyplot.plot(t,Error_rate,'r+')#plot
        print("Error rate itteration {} is {}% ".format(t, Error_rate))
        
    pyplot.ylim([0,50])
    pyplot.xlabel("Itterations")
    pyplot.ylabel("Error rate in %")
    pyplot.grid() #easier to visualize
    pyplot.show()
    
    return weights_classifier

adaboost_test = loadData("dataset/adaboost_test.csv")

weights = addaBoost(200, adaboost_train, adaboost_test)
