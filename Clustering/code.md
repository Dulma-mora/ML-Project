# Clustering

This is where the actual terror starts 😱💀

```
# pip install umap-learn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
    
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import random
np.random.seed(0)
```

In essence, because our target feature is Revenue, we are interested in dividing our data in two groups: buyers (1) and non-buyers (0) of online shops. 

### Reshaping the data

To be able to use clustering algorithms, we need to reshape our data. This is, to transform our data matrix so the algorithms can work and process it.

Creating a new function to plot 3D data

```
def plot3d(X, labels): # each point will be associated to a label
    # Set matplotlib to generate static images
    # %matplotlib inline # To make the plot iterative and explore the plot
    # Set matplotlib as interactive
    # This import is required to set up the 3D environment
    from mpl_toolkits.mplot3d import Axes3D
    
    unique_labels = np.unique(labels) #collecting all the unique labels
    # Convert negative values (outliers) into positive
    labels = labels.copy()
    labels += np.abs(unique_labels.min())
    nlabels = len(unique_labels)
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    # Define color mappings
    col = ('tab10' if nlabels <= 10 else 
          ('tab20' if nlabels <= 20 else
           'hsv'))
    cmap = plt.cm.get_cmap(col)
    smap = cm.ScalarMappable(
              norm=mcolors.Normalize(unique_labels.min(), unique_labels.max()+1), 
              cmap=cmap)
    # Plot the 3d coordinates (similar to standard scatter plots, we just
    # need to provide an additional z coordinate!)
    ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=labels, cmap=cmap)
    # Plot a color bar on the right
    plt.colorbar(mappable=smap, label='digit label', ticks=range(nlabels))
```

Reshaping:

```
dim = X_full_train.shape[1] # working on our previously defined object from the classification script

# Transform from pandas data frame to a numpy array
train_data_cluster = X_full_train.to_numpy()

# Reshape the data to allow dimensionality reduction
X_cluster = train_data_cluster.reshape(-1, dim)
print('X_cluster is a matrix of dimensions: {}'.format(X_cluster.shape))

y_cluster = y_full_train
print('y_cluster is a matrix of dimensions: {}'.format(y_cluster.shape))
```

---

## Dimensionality Reduction Analysis

#### PCA

```
from sklearn.decomposition import PCA

# defining a number of components
pca = PCA(n_components = 3)

X_prj = pca.fit_transform(X_cluster)
X_prj.shape
```

Using the function to plot the data in 3D:

```
plot3d(X_prj, labels = y_cluster)
```

> After seeing how the data is plotted we need to discuss it.


































































































































