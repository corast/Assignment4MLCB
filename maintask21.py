""" Task 2.1 KNN regression + classification """

import numpy as np
import pandas as pd
import bisect
from collections import Counter
import operator

def loadData(file):
    df = pd.read_csv(file, sep=',', header=0)
    return df.values

def knn(dataset, target, k):
    """ dataset with y column, target is target index, k is number of elements we return from sorted list by distance """
    distanceList = [] #Array keeping track of the distance of each element to target
    #Loop tru the array and calculate the distance to target.
    for i, data in enumerate(dataset):
        #We skip our target
        if(i == target):
            continue
        
        #Calculatre the distance between the two points
        distance = compare(data, dataset[target])
        touple = (distance, i)#Keep track of index in dataset and its distance to target.
        bisect.insort(distanceList, touple)
    
    #We only need to return the k first values.
    return distanceList[:k]

    #print("{} target 123".format(dataset[123]))
    #for datas in topk:
    #    print("{} index {} with a distance {}".format(dataset[datas[1]], datas[1] , datas[0]))

def compare(data, target):
    """ compute the distance to an input vector(list) to an index in the dataset as Euclidian distance """
    #sqrt ((xd-xt)^2+...+(xn-xt)^2)
    distance = 0
    #Itterate tru ever value expect the classification of the vectors
    for i in range(len(data)-1):
        distance +=(data[i]-target[i])**2 #(x_d-x_t)
    distance = np.sqrt(distance) #
    return distance

def regression(data, target, k):
    ypos = len(data[0])-1
    topk = knn(data, target, k)
    mean = 0
    for datas in topk:
        mean += data[datas[1]][ypos]
    mean = mean/k
    #Return the mean value.
    print("mean value for target data[{}]={} is {}".format(target,data[target],mean))

#Load the data for regression and find the value.
data = loadData('dataset/knn_regression.csv')
regression(data, 123, 10)

def knn_c(dataset, target, k):
    """ dataset with y column, target is target index, k is number of elements we return from sorted list by distance """
    distanceList = [] #Array keeping track of the distance of each element to target
    #Loop tru the array and calculate the distance to target.
    for i, data in enumerate(dataset):
        #Calculatre the distance between the two points
        distance = compare(data, target)
        touple = (distance, i)#Keep track of index in dataset and its distance to target.
        bisect.insort(distanceList, touple)
    
    #We only need to return the k first values.
    return distanceList[:k]

def classify(data, x, k):
    topk = knn_c(data, x, k)
    ypos = len(data[0])-1
    dictCounter = dict()
    for datas in topk:
        index = datas[1]
        dictCounter[data[index][ypos]] = dictCounter.get(data[index][ypos],0) + 1   
    print("{} classified as {}".format(x ,max(dictCounter.items(), key=operator.itemgetter(1))[0]))
    

#####Classification task################

#We need to change dataset, otherwise we wont match the length
data = loadData('dataset/knn_classification.csv')
classify(data,[6.3, 2.7, 4.91, 1.8],10)

