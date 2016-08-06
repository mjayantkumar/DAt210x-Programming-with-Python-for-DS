import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples = []
colors = []
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 
import os
import numpy as np

for img in os.listdir("Datasets/ALOI/32/"):
    if img.endswith(".png"):
        tmp = misc.imread("Datasets/ALOI/32/" + img)
        tmp[::2, ::2]
        X = (tmp / 255.0).reshape((-1,3))
      
        samples.append(X)
        colors.append('b')
#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 

for img in os.listdir("Datasets/ALOI/32i/"):
    if img.endswith(".png"):
        tmp = misc.imread("Datasets/ALOI/32i/" + img)
        tmp[::2, ::2]
        X = (tmp / 255.0).reshape((-1,3))

        colors.append('r')
        samples.append(X)
#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 
df = pd.DataFrame(samples[0])


#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
from sklearn import manifold

# train on input
iso = manifold.Isomap(n_neighbors=6, n_components=3)
iso.fit(df)

# transform input
M = iso.transform(df)


#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('2D')
ax.scatter(M[:,0], M[:,1], c=colors, marker='.', alpha=0.75)


#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_title('3D')
ax.scatter(M[:,0], M[:,1], M[:,2], c=colors, marker='.', alpha=0.75)


plt.show()

