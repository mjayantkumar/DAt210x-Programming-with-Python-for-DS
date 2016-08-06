import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import manifold

#Load up the /Module6/Datasets/parkinsons.data data set 
#into a variable X, being sure to drop the name column.
X = pd.read_csv('Datasets/parkinsons.data')

#Splice out the status column into a variable y and delete it from X.
y = X.loc[:, 'status']

X = X.drop(labels=['name','status'], axis=1)


'''Right after you splice out the status column, but before you process the 
train/test split, inject SciKit-Learn pre-processing code. 
Unless you have a good idea which one is going to work best, you're going to 
have to try the various pre-processors one at a time, checking to see if they 
improve your predictive accuracy.

Experiment with Normalizer(), MaxAbsScaler(), MinMaxScaler(), and StandardScaler().'''

T = preprocessing.StandardScaler().fit_transform(X)
#T = preprocessing.MinMaxScaler().fit_transform(X)
#T = preprocessing.MaxAbsScaler().fit_transform(X)
#T = preprocessing.Normalize(X)

#try experimenting with PCA n_component values between 4 and 14.

best_score = 0

for k in range(4, 14, 1):
    pca = PCA(n_components = k)
    pca.fit(T)
    pca_T = pca.transform(T)
    X_train, X_test, y_train, y_test = train_test_split(pca_T, y, 
                                                    test_size=0.30, 
                                                    random_state=7)
    
    for i in np.arange(0.05, 2, 0.05):
        for j in np.arange(0.001, 0.1, 0.001):
            svc = SVC(kernel = 'rbf', C = i, gamma = j)
            svc.fit(X_train, y_train)
            score = svc.score(X_test, y_test)
            if score > best_score:
                best_score = score

print ("Best Accuracy from PCA: ", best_score)

#run Isomap on the data
#Manually experiment with every inclusive combination of n_neighbors 
#between 2 and 5, and n_components between 4 and 6
iso = manifold.Isomap(n_neighbors=2, n_components=4)
iso.fit(T)
iso_T = iso.transform(T)
X_train, X_test, y_train, y_test = train_test_split(iso_T, y, 
                                                    test_size=0.30, 
                                                    random_state=7)

for i in np.arange(0.05, 2, 0.05):
    for j in np.arange(0.001, 0.1, 0.001):
        svc = SVC(kernel = 'rbf', C = i, gamma = j)
        svc.fit(X_train, y_train)
        score = svc.score(X_test, y_test)
        if score > best_score:
            best_score = score

print ("Best Accuracy from ISO: ", best_score) 

        