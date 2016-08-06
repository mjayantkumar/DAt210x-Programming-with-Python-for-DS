import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


#Load up the /Module6/Datasets/parkinsons.data data set 
#into a variable X, being sure to drop the name column.
X = pd.read_csv('Datasets/parkinsons.data')

#Splice out the status column into a variable y and delete it from X.
y = X.loc[:, 'status']

X = X.drop(labels=['name','status'], axis=1)

#Perform a train/test split. 30% test group size, with a random_state equal to 7.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=7)

#Create a SVC classifier. Don't specify any parameters, 
#just leave everything as default. Fit it against your training data
#and then score your testing data.

#What accuracy did you score?
svc = SVC()
svc.fit(X_train, y_train)
score = svc.score(X_test, y_test)

print ("Accuracy of the default SVC model : ", score) 



        