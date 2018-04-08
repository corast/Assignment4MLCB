"""Task 23 K-NN vs SVM vs Random Forest  """
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report  #confusion_matrix(y_true, y_predicted) return matrix
#k-NN
from sklearn.neighbors import KNeighborsClassifier
#SVM
from sklearn import svm
#Random Forest
from sklearn.ensemble import RandomForestClassifier  

digits = datasets.load_digits()
#Get the X and Y valuse from the dataset
X,Y = digits.get("data"), digits.get("target")
#Split the data into training and test set.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

target_names = [x for x in range(10)]

#k-NN
def knn_testk_values():
    for k in range(1,100,1):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train,Y_train)
        result = neigh.score(X_test, Y_test)
        print("k=%d accuaracy=%.2f%%" % (k, result*100 ))

neigh = KNeighborsClassifier()
neigh.fit(X_train,Y_train)
print("untuned RandomForest: accuaracy=%.2f%% " % (neigh.score(X_test, Y_test)*100) )
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train,Y_train)
print("tuned RandomForest: accuaracy=%.2f%% " % (neigh.score(X_test, Y_test)*100) )
predict_knn = neigh.predict(X_test)
cm_knn = confusion_matrix(Y_test,predict_knn)
cr_knn = classification_report(Y_test, predict_knn)#for debugging
#print(cr_knn)


#SVM
def svm_gamma():
    gammas = list(np.arange(0.0001, 0.0025, 0.0001))
    gammas.append(0.01)
    gammas.append(0.005)
    gammas.append(0.00001)
    for gamma in gammas:
        svmClassifier = svm.SVC(gamma=gamma)
        svmClassifier.fit(X_train,Y_train)
        result = svmClassifier.score(X_test, Y_test)
        print("gamma=%f accuaracy=%.2f%%" % (gamma, result*100 ))


svmClassifier = svm.SVC()
svmClassifier.fit(X_train, Y_train)
print("untuned SVM: accuaracy=%.2f%% " % (svmClassifier.score(X_test, Y_test)*100) )
svmClassifier = svm.SVC(gamma=0.001)
svmClassifier.fit(X_train, Y_train)
predict_svm = svmClassifier.predict(X_test)
print("tuned SVM: accuaracy=%.2f%% " % (svmClassifier.score(X_test, Y_test)*100) )
cm_svm = confusion_matrix(Y_test, predict_svm)
cr_svm = classification_report(Y_test, predict_svm)#for debugging

#RandomForest
def rf_n_trees():
    for n in range(5,105,5):
        randomForestClassifier = RandomForestClassifier(n_estimators=n)
        randomForestClassifier.fit(X_train,Y_train)
        result = randomForestClassifier.score(X_test, Y_test)
        print("n=%d accuaracy=%.2f%%" % (n, result*100 ))
#rf_n_trees()
randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X_train, Y_train)
print("untuned SVM: accuaracy=%.2f%% " % (randomForestClassifier.score(X_test, Y_test)*100) )
randomForestClassifier = RandomForestClassifier(n_estimators=30)
randomForestClassifier.fit(X_train, Y_train)
predict_rfc = randomForestClassifier.predict(X_test)
print("tuned RandomForest: accuaracy=%.2f%% " % (randomForestClassifier.score(X_test, Y_test)*100) )
cm_rfc = confusion_matrix(Y_test, predict_rfc)
cr_knn = classification_report(Y_test,predict_knn)#for debugging



def plot_confision_matrix(cm, classes, title='confusion matrix', cmap=plt.cm.Blues, normalized=False):
    """ Plot confusion matrix as either normalized or not. With coloring etc """
    if(normalized):
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    tresh = cm.max() / 2 #Tresh is halv of the value, aprox 25 in our case
    #chance text of the numbers if the square is a certain colour.(only for readability)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j], fmt),
            horizontalalignment='center',
            color="white" if cm[i,j] > tresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.figure()
#plt.subplot(1,3,1)
plot_confision_matrix(cm_knn, classes=target_names, title="Confusion matrix K-nn")

plt.figure()
#plt.subplot(1,3,2)
plot_confision_matrix(cm_svm, classes=target_names, title="Confusion matrix SVM")

plt.figure()
#plt.subplot(1,3,3)
plot_confision_matrix(cm_rfc, classes=target_names, title="Confusion matrix Random Forest")

plt.figure()
#plt.subplot(1,3,1)
plot_confision_matrix(cm_knn, classes=target_names, title="Normalized Confusion matrix K-nn",normalized=True)

plt.figure()
#plt.subplot(1,3,2)
plot_confision_matrix(cm_svm, classes=target_names, title="Normalized Confusion matrix SVM",normalized=True)

plt.figure()
#plt.subplot(1,3,3)
plot_confision_matrix(cm_rfc, classes=target_names, title="Normalized Confusion matrix Random Forest",normalized=True)

plt.show()


def plot():
    plt.imshow(digits.get("images")[2], cmap=plt.cm.Greys)
    plt.show()
    #images_labels = list(zip(digits.images, digits.target))

    #print(images_labels)