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

Using the function to inspect data:
May not be necessary (?)

```
X = 

plot3d(X, labels=y)
```

---

## Hierarchical Clustering

We are performing the Linkage Matrix method, so we need to construct it:

```
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate the linkage matrix
Z = linkage(X, method='ward', metric='euclidean') # TODO: defy X object
```

```
# Compute the linkages row numbers referring 
# to a merge with at least one cluster
mask = np.logical_or(Z[:,0] >= 1797, Z[:, 1]>= 1797) # TODO:
np.int32(Z[mask][:, [0, 1, 2, 3]])
np.where(mask)
```

#### Plotting the Dendogram

Defining function to plot the dendrogram

```
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(Z=None, model=None, X=None, **kwargs):
    annotate_above = kwargs.pop('annotate_above', 0)

    # Reconstruct the linakge matrix if the standard model API was used
    if Z is None:
        if hasattr(model, 'distances_') and model.distances_ is not None:
            # create the counts of samples under each node
            counts = np.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count

            Z = np.column_stack([model.children_, model.distances_,
                                              counts]).astype(float)
        else:
            Z = linkage(X, method=model.linkage, metric=model.affinity)
    
    if 'n_clusters' in kwargs:
        n_clusters = kwargs.pop('n_clusters')
        # Set the cut point just above the last but 'n_clusters' merge
        kwargs['color_threshold'] = Z[-n_clusters, 2] + 1e-6
        #kwargs['color_threshold'] = None
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    # Plot the corresponding dendrogram
    ddata = dendrogram(Z, ax=ax, **kwargs)
    
    # Annotate nodes in the dendrogram
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        nid = np.where(Z[:,2] == y)[0][0]
        if y > annotate_above:
            plt.plot(x, y, 'o', c=c)
            plt.annotate(str(nid-Z.shape[0]), (x, y), xytext=(0, -5),
                         textcoords='offset points',
                         va='top', ha='center')
    if kwargs['color_threshold']:
        plt.axhline(y=kwargs['color_threshold'], c='k')
    
    return fig, ax
   ```

See the graph:

```
# Plot the dendrogram, showing ony the ast 100 merges
# and cutting the dendrogram so that we obtain 10 clusters
plot_dendrogram(Z=Z, X=X,   # TODO: adjust X and Z to our data
                truncate_mode='lastp', 
                p=100, n_clusters=10)
```



































































































































