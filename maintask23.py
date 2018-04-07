"""Task 23 K-NN vs SVM vs Random Forest  """
import matplotlib.pyplot as plt
from sklearn import datasets
#k-NN
from sklearn.neighbors import KNeighborsClassifier
#SVM
from sklearn import svm
#Random Forest
from sklearn.ensemble import RandomForestClassifier  

digits = datasets.load_digits()
X,Y = digits.get("data"), digits.get("target")

print(X)
print(Y)

#images_labels = list(zip(digits.images, digits.target))

#print(images_labels)

#print(digits.images.shape)
#print(digits.get("images"))

plt.imshow(digits.get("images")[2], cmap=plt.cm.Greys)
plt.show()