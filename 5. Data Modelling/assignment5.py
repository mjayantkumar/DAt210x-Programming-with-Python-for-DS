import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

matplotlib.style.use('ggplot') # Look Pretty


def plotDecisionBoundary(model, X, y):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	padding = 0.6
	resolution = 0.0025
	colors = ['royalblue','forestgreen','ghostwhite']

 # Calculate the boundaries
	x_min, x_max = X[:, 0].min(), X[:, 0].max()
	y_min, y_max = X[:, 1].min(), X[:, 1].max()
	x_range = x_max - x_min
	y_range = y_max - y_min
	x_min -= x_range * padding
	y_min -= y_range * padding
	x_max += x_range * padding
	y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at said location
	xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
 # Plot the contour map
	plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)
	plt.axis('tight')

  # Plot our original points as well...
	for label in range(len(np.unique(y))):
		indices = np.where(y == label)
		plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

	p = model.get_params()
	plt.title('K = ' + str(p['n_neighbors']))


# 
# TODO: Load up the dataset into a variable called X. Check the .head and
# compare it to the file you loaded in a text editor. Make sure you're
# loading your data properly--don't fail on the 1st step!
#
# .. your code here ..
X = pd.read_csv("Datasets/wheat.data")
X.head()
#
# TODO: Copy the 'wheat_type' series slice out of X, and into a series
# called 'y'. Then drop the original 'wheat_type' column from the X
#
# .. your code here ..
y = X.loc[:, "wheat_type"]
X = X.drop(labels = ['id', 'wheat_type'], axis=1)

# TODO: Do a quick, "nominal" conversion of 'y' by encoding it to a SINGLE
# variable (e.g. 0, 1, 2). This is covered in the Feature Representation
# reading as "Method 1)". In actuality the classification isn't nominal,
# but this is the fastest way to encode you 3 possible wheat types into a
# label that you can plot distinctly. More notes about this on the bottom
# of the assignment.
#
# .. your code here ..
y=y.astype("category").cat.codes

#
# TODO: Basic nan munging. Fill each row's nans with the mean of the feature
#
# .. your code here ..
X.apply(lambda x: sum(x.isnull()), axis=0)

X.compactness = X.compactness.fillna(X.compactness.mean())
X.width = X.width.fillna(X.width.mean())
X.groove = X.groove.fillna(X.groove.mean())

# 
# TODO: Use SKLearn's regular "normalize" preprocessor to normalize X's feature data
#
# .. your code here ..
normalized_X = preprocessing.normalize(X)


#
# TODO: Project both your X_train and X_test features into PCA space.
# This has to be done because the only way to visualize the decision
# boundary in 2D, would be if your KNN algo ran in 2D as well
#
# .. your code here ..

pca = PCA(n_components = 2)
pca.fit(normalized_X)

X_pca = pca.transform(normalized_X)


#
# TODO: Split out your training and testing data.
# INFO: Use 0.33 test size, and use random_state=1. This is important
# so that your answers are verifiable. In the real world, you wouldn't
# specify a random_state.
#
# .. your code here ..

X_train, X_test, label_train, label_test = train_test_split(X_pca, y, test_size=0.33, random_state=1)

#
# TODO: Run KNeighborsClassifier. Start out with K=7 neighbors. NOTE:
# Be sure train your classifier against the PCA transformed feature
# data above! You do not, however, need to transform your labels.
#
# .. your code here ..
knn=KNeighborsClassifier(n_neighbors =1 )
knn.fit(X_train, label_train)

# HINT: Ensure your KNeighbors classifier object from earlier is called 'knn'.
# This method plots your TEST points against the boundary learned from your
# training data:
plotDecisionBoundary(knn, X_test, label_test)


#
# TODO: Display the accuracy score.
#
# NOTE: You don't have to run .predict before calling .score, since
# .score will take care of running your predictions for the params
# you provided.
#
# .. your code here ..
knn.score(X_test, label_test)

#
# BONUS: Instead of the ordinal conversion, try and get this assignment
# working with a proper Pandas get_dummies for feature encoding. HINT:
# You might have to update some of the plotDecisionBoundary code.


plt.show()

