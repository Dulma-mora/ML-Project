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

> Explanation of why we need to use dimensionality reduction algorithms

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

---

# Clustering algorithms

During this project, we are going to perform three clustering algorithms and compare them with each other:
- Agglomerative Clustering Algorithm (Hierarchical Clustering)
- KNN Algorithm
- DBSCAN Algorithm

---

## Agglomerative Clustering Analysis (Hierarchical Clustering)

#### Preparing the previsualization Dendogram

**Linkage Matrix**

> Discuss what is a linkage matrix and why are we creating it, also discuss what are we gonna use it for.

The importance of defining the linkage matrix since the beggining is because the agglomerative clustering algorithm uses it to obtain the dendogram. 

```
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate the linkage matrix
Z = linkage(X_cluster, method='ward', metric='euclidean')
```

```
# Compute the linkages row numbers referring
# to merge with at least one cluster
mask = np.logical_or(Z[:,0] >= 1797, Z[:, 1]>= 1797)
np.int32(Z[mask][:, [0, 1, 2, 3]])
np.where(mask)
```

#### Plotting the Previsualization Dendrogram

For this clustering method, we need to define a function that allows us to plot the dendrogram of every analysis.     
Here, we are defining a function to plot all dendrograms. 

```
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(Z=None, model=None, X=None, **kwargs):
    annotate_above = kwargs.pop('annotate_above', 0)

    # Reconstruct the linkage matrix if the standard model API was used
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
In addition, this first dendrogram generation is a static display that allows an overview of how data points are grouped and related to each other as clusters are merged. **A specific number of clusters is not automatically selected at this point even though we are performing an agglomerative clustering algorithm**.      
This dendrogram provides valuable information about the clustering hierarchy and can help determine an approximate number of clusters.

```
# Plot the dendrogram, showing ony the ast 100 merges
# and cutting the dendrogram so that we obtain 10 clusters
plot_dendrogram(Z=Z, X=X_cluster,
                truncate_mode='lastp', 
                p=100, n_clusters=10)
```

#### Auxiliary functions 

```
# Recursively backtrack the dendrogram, collecting
# the list of sample id starting from an initial point
def get_node_leaves(Z, idx, N):
    n1, n2 = int(Z[idx,0]), int(Z[idx,1])
    leaves = []
    for n in [n1, n2]:
        if n < N:
            leaves += [n]
        else:
            leaves += get_node_leaves(Z, n-N, N)
    return leaves

# Plot a number of images (at most maxn) under a cluster/sample id
def plot_node(Z, X, y, idx, maxn=15*15):
    leaves = get_node_leaves(Z, idx, X.shape[0])
    labels, counts = np.unique(y[leaves], return_counts=True)
    nleaves = len(leaves)
    print(pd.DataFrame(np.array(counts).reshape(1,-1), 
                       columns=labels, index=["Frequency:"]))
    print("Images in the cluster:", len(leaves), "/", X.shape[0])

    random.shuffle(leaves)
    leaves = leaves[:maxn]
    h = min((nleaves // 15)+1, 15)
    w = nleaves if nleaves < 15 else 15
    
    fig, axes = plt.subplots(h, w, figsize=(w, h),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

    # For each subfigure (from 0 to 100 in the 10x10 matrix)
    for i, ax in enumerate(axes.flat):
        if i < nleaves:
            ax.imshow(X[leaves[i]].reshape(8, 8), cmap='binary', interpolation='nearest')
            ax.text(0.05, 0.05, str(y[leaves[i]]), transform=ax.transAxes, color='r')
        else:
            ax.set_axis_off()
```

Plotting the first node:

```
# Plot the first node
# Remember: we expect only two samples,
# the most similar ones in the dataset!
plot_node(Z, X_cluster, y_cluster, -11)
```

### Linkage Methods of Agglomerative Clustering Algorithm 

Agglomerative Hierarchical Clustering Algorithm is a general clustering approach that involves the iterative union of clusters to form a hierarchy of clusters. During this process, different linkage methods are used, which are criteria to calculate the distance or similarity between clusters.

Here, we are comparing five different methods: single, average, complete, centroid and ward.
> Write more about them

And we are going to select the best method for our model.

```
# defyning methods
methods = ['single', 'average', 'complete', 'centroid', 'ward']

# We are gonna get one plot for every method (5)
for method in methods:
    Z = linkage(X, method=method, metric='euclidean') # X_cluster ?
    fig, ax = plot_dendrogram(Z=Z, X=X, truncate_mode='lastp', 
                              p=100, n_clusters=10)
    ax.set_title(method)
```

> Discussing which one was the best one and why
> Then we need to add more n values to that best method
> Then visualize it (in 3D or 2D, depends on what we conclude)

```
# here we add more n values to the best method
#
#
```

### Agglomerative Clustering Algorithm using the () method

Now that we dentified the best method, we are using it as a parameter to perform the agglomerative algorithm:

```
from sklearn.cluster import AgglomerativeClustering

distance_threshold=None #
n_clusters=10

model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')

y_predict = model.fit_predict(X_cluster)

plot3d(X_cluster, labels=y_predict)
plot_dendrogram(model=model, X=X_cluster, truncate_mode='lastp', p=100,
            n_clusters=n_clusters,
            color_threshold=distance_threshold)
```
The output of the previous code is a 3D plot and a dendrogram.

### Clustering Metrics

Now that we obtained an output, our aim is to evaluate the quality of the cluster and find the optimal number of clusters for the analyzed data.    

**WSS (Within-cluster Sum of Squares):** Measures the sum of the squares of the distances between each point and the centroid of its cluster. The smaller the value, the more compact the clusters will be.    
**BSS (Between-cluster Sum of Squares):** Measures the sum of the squares of the distances between the centroids of the clusters and the global centroid of the data. The higher the value, the further apart the clusters will be.
**Silhouette Score:** Calculates how well each sample does within its own cluster compared to other clusters. A value closer to 1 indicates that the samples are well separated in their clusters.     
**Correlation:** Calculates the correlation between the incidence matrix and the similarity matrix. The higher the value, the better the cluster structure


**Calculation of the Clustering Metrics**

```
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def incidence_mat(y_pred):
 npoints = y_pred.shape[0]
 mat = np.zeros([npoints, npoints])
 # Retrieve how many different cluster ids there are
 clusters = np.unique(y_pred)
 nclusters = clusters.shape[0]

 for i in range(nclusters):
 sample_idx = np.where(y_pred == i) #indices of the samples in this cluster
 # Compute combinations of these indices
 xx, yy = np.meshgrid(sample_idx, sample_idx)
 mat[xx, yy] = 1

 return mat
def similarity_mat(X, metric):
 dist_mat = pairwise_distances(X, metric=metric)
 min_dist, max_dist = dist_mat.min(), dist_mat.max()

 # Normalize distances in [0, 1] and compute the similarity
 sim_mat = 1 - (dist_mat - min_dist) / (max_dist - min_dist)
 return sim_mat
def correlation(X, y_pred, metric):
 inc = incidence_mat(y_pred)
 sim = similarity_mat(X, metric)

 # Note: we can eventually remove duplicate values
 # only the upper/lower triangular matrix
 # triuidx = np.triu_indices(y_pred.shape[0], k=1)
 # inc = inc[triuidx]
 # sim = sim[triuidx]

 inc = normalize(inc.reshape(1, -1))
 sim = normalize(sim.reshape(1, -1))
 corr = inc @ sim.T

 return corr[0,0]
```
Testing:

```
correlation(X_cluster.reshape(-1, 64), y_predict, 'euclidean') # using the predicted labels
```

**Comparing with a random clustering**

```
y_rand = np.random.randint(0, 10, y.shape[0])
correlation(X_cluster, y_rand, 'euclidean')
```
To inspect the similarity matrix and take a look at how many points within the same cluster are distant from
each other we can resort to a sorted similarity matrix.

```
def sorted_mat(sim, y_pred):
 idx_sorted = np.argsort(y_pred)
 # Sort the rows
 sim = sim[idx_sorted]
 # Sort the columns
 sim = sim[:, idx_sorted]

 return sim
def plot_sorted_mat(sim, y_pred):
 sim = sorted_mat(sim, y_pred)

 fig, ax = plt.subplots(figsize=(40,30))
 ax = sns.heatmap(sim, ax=ax)
 # Remove ruler (ticks)
 ax.set_yticks([])
 ax.set_xticks([])
```

**Visualizing the Ordered Similarity Matrix and the Incidence Matrix**

```
# Try to select different distances!
sim = similarity_mat(X_cluster, metric='euclidean')
# plot sorted ...
plot_sorted_mat(sim, y_predict)
```

```
# Plot the sorted incidence matrix and compare it with the similarity matrix
inc = incidence_mat(y_predict)
plot_sorted_mat(inc, y_predict) # using predefined functions
```

#### Deciding the number of clusters in the dendrogram

> This part is sospechosa, we need to understand why they put the values on zero again, maybe we need to do something different

```
Z = linkage(X_cluster, metric='euclidean', method='ward')
fig, ax = plot_dendrogram(Z=Z, X=X_cluster, truncate_mode='lastp',
 p=100, n_clusters=0)
```

**Elbow analysis**

Now, here we are going to use elbow analysis to approximate the number of clusters that our data can be split into.

> Write about BSS, BSS, silhouette score and why we need those metrics

```
def wss(X, y_pred, metric):
    # Compute the incidence matrix
    inc = incidence_mat(y_pred)
    # Compute the distances between each pair of nodes
    dist_mat = pairwise_distances(X, metric=metric)
    # Use the incidence matrix to select only the 
    # distances between pair of nodes in the same cluster
    dist_mat = dist_mat * inc
    # Select the lower/upper triangular part of the matrix
    # excluding the diagonal
    triu_idx = np.triu_indices(X.shape[0], k=1)
    
    wss = (dist_mat[triu_idx] ** 2).sum()
    
    return wss

def bss(X, y_pred, metric):
    # Compute the incidence matrix
    inc = incidence_mat(y_pred)
    # Compute the distances between each pair of nodes
    dist_mat = pairwise_distances(X, metric=metric)
    # Use the incidence matrix to select only the 
    # distances between pair of nodes in different clusters
    dist_mat =  dist_mat * (1 - inc)
    # Select the lower/upper triangular part of the matrix
    # excluding the diagonal
    triu_idx = np.triu_indices(X.shape[0], k=1)
    
    bss = (dist_mat[triu_idx] ** 2).sum()
    
    return bss

print("WSS", wss(X_cluster, y_predict, 'euclidean'))
print("BSS", bss(X_cluster, y_predict, 'euclidean'))
```

Elbow plot

```
from sklearn.metrics import silhouette_score

wss_list, bss_list, sil_list = [], [], []
clus_list = list(range(1, 15))

for nc in clus_list:
    model = AgglomerativeClustering(n_clusters=nc,
                                    affinity='euclidean', 
                                    linkage='ward')

    y_predict = model.fit_predict(X)
    
    wss_list.append(wss(X_cluster, y_predict, 'euclidean'))
    bss_list.append(bss(X_cluster, y_predict, 'euclidean'))
    if nc > 1:
        sil_list.append(silhouette_score(X_cluster, y_predict, metric='euclidean'))
    
plt.plot(clus_list, wss_list, label='WSS')
plt.plot(clus_list, bss_list, label='BSS')
plt.legend()
plt.show()

plt.plot(clus_list[1:], sil_list, label='Average silhuette score')
plt.legend()
```

#### Splitting the clusters

Now, because (number) was the result of the plots, we are assigning (number) clusters in total.

```
Z = linkage(X_cluster, metric='euclidean', method='ward')
fig, ax = plot_dendrogram(Z=Z, X=X_cluster, truncate_mode='lastp', 
                          p=100, n_clusters=9) # CHANGE THIS VALUE
```

### Precision, Recall, and Purity

> Explain what are these things and that they still are cluster evaluating metrics

These metrics are used to evaluate the performance of the clustering performed. For all of them, a value closer to 1 indicates better performance of the clustering algorithm.
A value of 1 for Precision and Recall indicates a perfect assignment of instances to a specific cluster and perfect identification of all instances in a cluster, respectively.
A Purity value of 1 indicates that the generated clusters perfectly match the true labels.

```
def get_Ncounts(y_predict, y_true, k, j=None):
    N = y_true.shape[0]
    Nk_mask = y_predict == k
    Nk = Nk_mask.sum()
    Nj, Nkj = None, None
    if j is not None:
        Nj_mask = y_true == j
        Nj = Nj_mask.sum()
        Nkj = np.logical_and(Nj_mask, Nk_mask).sum()
    return N, Nk, Nj, Nkj

def precision(y_predict, y_true, k, j):
    N, Nk, Nj, Nkj = get_Ncounts(y_predict, y_true, k, j)
    return Nkj / (Nk + 1e-8)
    
def recall(y_predict, y_true, k, j):
    N, Nk, Nj, Nkj = get_Ncounts(y_predict, y_true, k, j)
    return Nkj / (Nj + 1e-8)

def F(y_predict, y_true, k, j):
    p = precision(y_predict, y_true, k, j)
    r = recall(y_predict, y_true, k, j)
    return (2*p*r) / (p+r)

def purity(y_predict, y_true, k):
    cls = np.unique(y_true)
    prec = [precision(y_predict, y_true, k, j) for j in cls]
    return max(prec)

def tot_purity(y_predict, y_true):
    N = y_true.shape[0]
    nc = len(np.unique(y_true))
    p = 0
    for k in range(nc):
        N, Nk, _, _ = get_Ncounts(y_predict, y_true, k)
        pk = purity(y_predict, y_true, k)
        p += (Nk / N) * pk
    return p
```

**Calculating the Total Purity of the cluster**

```
tot_purity(y_predict, y_cluster)
```

---

## K Means Algorithm

Different initialization methods

```
# Random
model = KMeans(n_clusters=2, init="random", random_state=0)             
print("Random Kmeans purity", tot_purity(model.fit_predict(X_cluster), y_cluster))

# K-Means++
model = KMeans(n_clusters=2, init="k-means", random_state=0)             
print("Kmeans++ purity", tot_purity(model.fit_predict(X_cluster), y_cluster))

# Hierarchical
model = AgglomerativeClustering(n_clusters=2, 
                                 distance_threshold=distance_threshold, 
                                 affinity='euclidean', linkage='complete')
y_predict = hmodel.fit_predict(X_cluster)

### TODO: figure out what to put in these two lines
#centroids = np.stack([... for k in range(10)]) # what do we need to write here?
#model = KMeans(n_clusters=2, init=..., n_init=1, random_state=0)             
print("Hierarchical+Kmeans purity", tot_purity(model.fit_predict(X_cluster), y_cluster))

plot3d(X_cluster, labels=y_predict)
```

> Discuss the results

---

## DBSCAN

This algorithm is based on density and core points.

> More description of the DBSCAN algorithm

The algorithm starts with a random point p and retrieves all the points which are **density reachable** from p. If p is a core point the cluster is kept and the algorithm moves to the next unvisited point until all points have bees visited.

```
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

model = DBSCAN(eps=20, min_samples=10)
y_predict = model.fit_predict(X_cluster)
print("DBSCAN purity", tot_purity(y_predict, y_cluster))
```

```
print("Number of outliers", (y_predict == -1).sum())
ids, counts = np.unique(y_predict, return_counts=True)
print(pd.DataFrame(counts.reshape(1,-1), columns=ids, index=['']))


plot3d(X, labels=y_predict)
```

#### Deciding Eps and MinPts values

> Discuss something about all this concepts

Creating a function to estimate metrics
```
def make_scorer(metric):
    def scorer(estimator, X, y):
        y_pred = estimator.fit_predict(X)
        return metric(y_pred, y)
    return scorer
```

Using it

```
from sklearn.model_selection import GridSearchCV

params = {'eps': [20], 'min_samples': range(5,20)} # how to select the range?
cv = GridSearchCV(model, params, scoring=make_scorer(tot_purity), cv=3)
cv = cv.fit(X_cluster, y_cluster)
```

```
print(cv.best_params_)
print("CV score", tot_purity(cv.best_estimator_.fit_predict(X_cluster), y_cluster))

pd.DataFrame(cv.cv_results_)
```

---

# Algorithm Comparison

```
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
Xd, yd = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(Xd, transformation)
aniso = (X_aniso, yd)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
    (aniso, {'eps': .15, 'n_neighbors': 2,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (blobs, {}),
    (no_structure, {})]
```

```
plt.figure(figsize=(20, 20))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    Xd, yd = dataset

    # normalize dataset for easier parameter selection
    Xd = StandardScaler().fit_transform(Xd)

    # ============
    # Create cluster objects
    # ============
    clustering_algorithms = [
        ('KMeans', cluster.KMeans(n_clusters=params['n_clusters'])),
        ('Hierarchical (avg)', cluster.AgglomerativeClustering(
                               n_clusters=params['n_clusters'], linkage='average')),
        ('Hierarchical (single)', cluster.AgglomerativeClustering(
                                  n_clusters=params['n_clusters'], linkage='single')),
        ('Hierarchical (complete)', cluster.AgglomerativeClustering(
                                    n_clusters=params['n_clusters'], linkage='complete')),
        ('Hierarchical (ward)', cluster.AgglomerativeClustering(
                                    n_clusters=params['n_clusters'], linkage='ward')),
        ('DBSCAN', DBSCAN(eps=0.15, min_samples=3)),
    ]

    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        algorithm.fit(Xd)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(Xd)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(Xd[:, 0], Xd[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()
```





























