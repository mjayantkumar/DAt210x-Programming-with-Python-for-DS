import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv("Datasets/wheat.data", index_col=0, header=0)

#
# TODO: Create a 2d scatter plot that graphs the
# area and perimeter features
# 
# .. your code here ..
s1 = df[['area', 'perimeter']]
s1.plot.scatter('area', 'perimeter', marker = 'o')

#
# TODO: Create a 2d scatter plot that graphs the
# groove and asymmetry features
# 
# .. your code here ..
s2 = df[['groove', 'asymmetry']]
s2.plot.scatter('groove', 'asymmetry', marker = '.')

#
# TODO: Create a 2d scatter plot that graphs the
# compactness and width features
# 
# .. your code here ..
s3 = df[['compactness', 'width']]
s3.plot.scatter('compactness', 'width', marker = '^')


# BONUS TODO:
# After completing the above, go ahead and run your program
# Check out the results, and see what happens when you add
# in the optional display parameter marker with values of
# either '^', '.', or 'o'.


plt.show()


