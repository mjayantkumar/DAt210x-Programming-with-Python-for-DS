#
# This code is intentionally missing!
# Read the directions on the course lab page!
#

import pandas as pd
import matplotlib.pyplot as plt


import matplotlib

from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import parallel_coordinates

# Look pretty...
matplotlib.style.use('ggplot')

df = pd.read_csv("Datasets/wheat.data", index_col=0, header=0)

# create and show andrews curve plot
plt.figure()
andrews_curves(df, 'wheat_type')
plt.show()


# dropping 'area' and 'perimeter' features
df = df.drop('area', axis=1)
df = df.drop('perimeter', axis=1)


# create and show andrews curve plot on dropped dataset
plt.figure()
andrews_curves(df, 'wheat_type')
plt.show()