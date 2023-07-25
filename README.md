# ML-Project
A few codes for our Machine Learning Project


## Logistic Regression Code

```
from sklearn.model_selection import train_test_split

# X, a matrix that contains all the observations for each predictor
X_features = data_training.drop(['Revenue'], axis = 1).columns
X = data_training[X_features].to_numpy()  
   
# y, a vector of the outputs
y = data_training['Revenue'].to_numpy()  

# Defining the size and the seed for the validation set
split_seed = 42
split_test_size = 0.2

# Split X and y into train and validation set
X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X, 
                                                  y, 
                                                  test_size=split_test_size, 
                                                    random_state=split_seed)

# print train set
print("X_cl_train is a matrix of dimensions: {X_cl_train.shape}"
print("y_cl_train is a vector of dimensions: {y_cl_train.shape}"

# print validation set
print("X_cl_val is a matrix of dimensions: {X_cl_val.shape}"
print("y_cl_val is a vector of dimensions: {y_cl_val.shape}"

X = X_cl_train.to_numpy()
X_t = X_cl_test.to_numpy()

from sklearn.linear_model import LogisticRegression             # 1- model selection
model = LogisticRegression(solver="newton-cg", penalty='none')  # 2- hyperparams
model.fit(X, y_cl_train)

# Print the estimated coefficients
print("beta0 =", model.intercept_.squeeze())
print("beta1 =", model.coef_.squeeze())

print("Train accuracy:", accuracy(y_train, model.predict(X)))
print("Test accuracy:", accuracy(y_cl_test, model.predict(X_t)))

```
---

## Multiple Logistic Regression

```
# Train a logistic regressor that uses all features
X_full_feat = data_training.drop(['Revenue'], axis = 1).columns
X_full_train = data_training[X_full_feat]

# y full train
y_full_train = data_training["Revenue"].to_numpy()

# Defining the size and the seed 
split_seed = 42
split_test_size = 0.2


X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X_full_train, y_full_train,
                                                    test_size=split_test_size, 
                                                    random_state=split_seed)

# print train set
print("X_cl_train is a matrix of dimensions: {X_cl_train.shape}"
print("y_cl_train is a vector of dimensions: {y_cl_train.shape}"

# print validation set
print("X_cl_val is a matrix of dimensions: {X_cl_val.shape}"
print("y_cl_val is a vector of dimensions: {y_cl_val.shape}"
```

```
from sklearn.linear_model import LogisticRegression            # 1- model selection
model = LogisticRegression(solver='newton-cg', penalty='none') # 2- hyperparams
model.fit(X_cl_train, y_cl_train)                               # 3- model fitting
y_predict = model.predict(X_cl_train)                        # 4- model testing

print("Train accuracy:", accuracy(y_cl_train, model.predict(X_cl_train)))
print("Test accuracy:", accuracy(y_cl_test, model.predict(X_cl_test)))

# z test table
z_test(X_cl_train, y_cl_train, model, ["Intercept", *X_full_feat], alpha=0.005)
```

#### Feature selection
```
# ===================================
#   Code from Lab03.01 - 05/05/2021
# 03.01.ScikitFeaturesSelection.ipynb
# ===================================

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

def get_evaluator(scorer):
    def evaluator(model, X, y, trained=False):
        if not trained:
            model = model.fit(X, y)
        score = scorer(model, X, y)
        return model, score
    return evaluator   

def get_cv_evaluator(scorer, cv=3):
    def evaluator(model, X, y, trained=False):            
        scores = cross_val_score(model, X, y, scoring=scorer, cv=cv)
        if not trained:
            model = model.fit(X, y)
        return model, np.mean(scores)
    
    return evaluator

def get_val_evaluator(scorer, val_size=0.1):
    def evaluator(model, X, y, trained=False):
        X_train_small, X_val, y_train_small, y_val = train_test_split(X, y, 
                                                                      test_size=val_size,
                                                                      random_state=0)
        
        if not trained:
            model = model.fit(X_train_small, y_train_small)
        score = scorer(model, X_val, y_val) 
        
        return model, score
    
    return evaluator

def forward_selection(Xtrain_pd, ytrain, Xtest_pd, ytest,
                      candidates_evaluator, candidates_argbest, # Metric to be used at 2.b
                      subsets_evaluator, subsets_argbest,       # Metric to be used at 3
                      test_evaluator=None, test_argbest=None,
                      candidates_scorer_name=None,  # Name of 2. figure
                      subsets_scorer_name=None,     # Name of 3. figure
                      verbose=True, weight_step3=0):   
    test_evaluator = subsets_evaluator if not test_evaluator else test_evaluator
    test_argbest = subsets_argbest if not test_argbest else test_argbest
    
    # Global variable init
    # ====================
    num_features = Xtrain_pd.shape[-1]
    best_candidate_metric = []
    # subsets_* are lists containing one value for each Mk model (the best of the Mk candidates)
    subsets_test = []
    subsets_metric = []        # The best metric of each subset of dimension 'dim'
    subsets_best_features = [] # The best features combination in each subset of dimension 'dim'
    # A figure to keep track of candidates scores in each Mk subset
    plt.figure()
    candidate_fig = plt.subplot(111) # A global matplotlib figure
    num_evaluations = 0        # A conter to keep track of the total number of trials
    
    selected_features = []
    all_features = Xtrain_pd.columns
    
    
    # 1. Train M0
    # ===========
    model = DummyRegressor()
    # Compute (2.b) metrics
    model, score = candidates_evaluator(model, Xtrain_pd[[]], ytrain)
    best_candidate_metric.append(score)
    subsets_best_features.append([])
    _ = candidate_fig.scatter([0], [score], color="b")
    # Compute metric for step 3.
    _, score = subsets_evaluator(model, Xtrain_pd[[]], ytrain, trained=True)
    subsets_metric.append(score)
    _, score = test_evaluator(model, Xtrain_pd[[]], ytrain, trained=True)
    subsets_test.append(score)
    
    # 2. Evaluate all Mk candidates with
    #    k=0...P features
    # =========================================
    for dim in range(num_features):
        candidate_metrics = [] # Keep track of candidates metrics. Will be used to select the best
        candidate_models = []  # Keep track of candidates trained models
        
        # 2.a Fixed the number of features 'dim', look at
        #     all the possible candidate models with that
        #     cardinality
        # ===============================================
        remaining_features = all_features.difference(selected_features)
        
        for new_column in remaining_features:
            Xtrain_sub = Xtrain_pd[selected_features+[new_column]].to_numpy()
            # //==========================================\\
            # || ***** Difference from previous lab ***** ||
            # \\==========================================//
            model = LogisticRegression(solver="newton-cg", penalty='none')
            model, score = candidates_evaluator(model, Xtrain_sub, ytrain)
            candidate_models.append(model)
            candidate_metrics.append(score)
            num_evaluations += 1
            
        _ = candidate_fig.scatter([Xtrain_sub.shape[-1]]*len(candidate_metrics), candidate_metrics,
                                  color="b")
            
        # 2.b Select the best candidate among those using
        #     the same number of features (2.a)
        # ===============================================
        idx_best_candidate = candidates_argbest(candidate_metrics)
        # Update selected feature
        selected_features.append(remaining_features[idx_best_candidate])
        # Save best candidate features
        best_candidate_metric.append(candidate_metrics[idx_best_candidate])
        best_features = selected_features.copy()
        subsets_best_features.append(best_features)
        
        
        # Compute metric for step 3.
        best_subset_model = candidate_models[idx_best_candidate]
        best_subset_Xtrain = Xtrain_pd[best_features].to_numpy()
        best_subset_Xtest = Xtest_pd[best_features].to_numpy()
        _, score = test_evaluator(best_subset_model, best_subset_Xtest, ytest, trained=True)
        subsets_test.append(score)
        _, score = subsets_evaluator(best_subset_model, best_subset_Xtrain, ytrain, trained=True)
        subsets_metric.append(score)
        num_evaluations += weight_step3 
        
        if verbose:
            print("............")
            print("Best model (M{}) with {} features: {}".format(dim+1, dim+1, best_features))
            print("M{} subset score (3.): {}".format(dim+1, score))
        
    # 3. Among all best candidates with increasing number
    #    of features, select the best one
    # ===================================================
    best_subset_idx = subsets_argbest(subsets_metric)
    best_features = subsets_best_features[best_subset_idx]
    
    if verbose:
        print("\n Best configuration has {} features".format(best_subset_idx))
        print("Features: {}".format(subsets_best_features[best_subset_idx]))
        print("Total number of trained models:", num_evaluations)
    
    # Complete the subsets_fig figure by plotting
    # a line connecting all best candidate score
    best_candidate_score_idx = candidates_argbest(best_candidate_metric)
    _ = candidate_fig.plot(range(len(best_candidate_metric)), best_candidate_metric)
    _ = candidate_fig.scatter(best_candidate_score_idx, best_candidate_metric[best_candidate_score_idx],
                              marker='X', label="Best", color="r")
    candidate_fig.set_title(candidates_scorer_name)
    candidate_fig.legend()
    
    # Plot a figure to show how te 3. metric evolves
    plt.figure()
    subsets_fig = plt.subplot(111)
    _ = subsets_fig.plot(range(len(subsets_metric)), subsets_metric, label="Selection (3.) scores")
    _ = subsets_fig.scatter(best_subset_idx, subsets_metric[best_subset_idx],
                              marker='X', label="Best (3.) score", color="r")
    best_test_score_idx = test_argbest(subsets_test)
    _ = subsets_fig.plot(range(len(subsets_test)), subsets_test, label="Test scores")
    _ = subsets_fig.scatter(best_test_score_idx, subsets_test[best_test_score_idx],
                              marker='X', label="Best test score", color="y")
    subsets_fig.set_title(subsets_scorer_name)
    subsets_fig.legend()
```

```
# change this (?) if doesnt work
cv = 10
forward_selection(X_cl_train, y_cl_train, X_cl_test, y_cl_test,
                  get_evaluator(make_scorer(accuracy)), np.argmax, # 2.
                  get_cv_evaluator(make_scorer(accuracy), cv), np.argmax, # 3.
                  get_evaluator(make_scorer(accuracy)), np.argmax, # test
                  candidates_scorer_name="Accuracy",
                  subsets_scorer_name="Accuracy (CV)",
                  verbose=True, weight_step3=cv)
```
---

## Linear Discriminant Analysis

```
# doing weird things
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
model = LDA(store_covariance = TRUE)
model = model.fit(X_cl_train, y_cl_train)

print("Train accuracy:", accuracy(y_cl_train, model.predict(X_cl_train)))
print("Test accuracy:", accuracy(y_cl_test, model.predict(X_cl_test)))
```
IGNORE THIS CODE
```
# what is this shit for????

split_seed = 42
split_test_size = 0.2


X_lda_train, X_lda_test, y_lda_train, y_lda_test = train_test_split(X_full_train, y_full_train,
                                                    test_size=split_test_size, 
                                                    random_state=split_seed)

# --- IGNORE
#X_feat = ['typea', 'age']
#X = pd_data[X_feat]
#X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                  #  test_size=split_test_size, 
                                                   # random_state=split_seed)

model = LDA(store_covariance=True)
model = model.fit(X_lda_train, y_lda_train)

print("Train accuracy:", accuracy(y_lda_train, model.predict(X_lda_train)))
print("Test accuracy:", accuracy(y_lda_test, model.predict(X_da_test)))
```



```
from math import atan, degrees
from matplotlib import colors
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches

def plot_gaussian(model, mu, covar, x, y=None, n_stdev_bands=2):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if covar.ndim == 2:
        covar = np.repeat(covar[None], mu.shape[0], axis=0)
        
    # Confidence Ellipses
    # ===================
    
    ax = plt.subplot(111, aspect='equal')
    legend = []
    # We plot the distributions only if we are
    # using just 2 features (see next section)
    if mu.shape[-1] == 2:
        # 1- Compute the rotation and the variance 
        #    in the new rotated coordinate system
        # ========================================
        variance, transforms = np.linalg.eigh(covar)
        # 2- Compute the standard deviations
        # ==================================
        stdev = np.sqrt(variance)
        
        max_x, min_x, max_y, min_y = 0.0, 0.0, 0.0, 0.0
        cmap = cm.rainbow(np.linspace(0, 1, mu.shape[0]))
        for mean, stdev, transform, color in zip(mu, stdev, transforms, cmap):
            # We collect some labels to print in the legend
            legend += [mpatches.Patch(color=color,
                                      label=f'mu {mean[0]:.2f}, {mean[1]:.2f} '
                                      f'sigma {stdev[0]:.2f} {stdev[1]:.2f}')]
            # We compute several confidence ellipses
            # multiple of the standard deviation
            for j in range(1, n_stdev_bands + 1):
                # 3- Plot the ellipses
                # ==================================
                ell = mpatches.Ellipse(xy=(mu[0], mu[1]), # The center position (xc, yc)
                                       # The width and height:
                                       # axis-aligned standard deviation (multiple of 2*j)
                                       width=stdev[0] * j * 2, height=stdev[1] * j * 2,
                                       # The rotation angle degree(atan(v1/v0))
                                       angle=degrees(atan(transform[0, 1] / transform[0, 0])),
                                       # Color properties and transparency
                                       alpha=1.0, edgecolor=color, fc='none')
                ax.add_artist(ell)
                # Compute the picture size based on how many bands
                # we plotted
                max_x = max(max_x, mean[0] + stdev[0] * j * 1.5)
                max_y = max(max_y, mean[1] + stdev[1] * j * 1.5)
                min_x = min(min_x, mean[0] - stdev[0] * j * 1.5)
                min_y = min(min_y, mean[1] - stdev[1] * j * 1.5)
    else:
        # Otherwise just compute the plot boundaries
        max_x, min_x = x[:, 0].max()+x[:, 0].ptp()/2, x[:, 0].min()-x[:, 0].ptp()/2
        max_y, min_y = x[:, 1].max()+x[:, 1].ptp()/2, x[:, 1].min()-x[:, 1].ptp()/2
        
    # Boundaries 
    # ==========
    # A part from the actual boundary, we also want to
    # color the space regions with the same color as the
    # predicted class!
    # Given the picture size x \in [min_x, max_x], y \in [min_y, max_y]
    # we sample the space at regular intervals and create a fake dataset
    # with features that cover all the space.
    # The meshgrid function takes two array (a ruler on the two axis),
    # combine them and return two matrices of shape [y_size, x_size]
    # containing all the x coordinates (the first) and all the y coordinates
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 1000),
                         np.linspace(min_y, max_y, 1000))
    # xx.shape = [1000, 1000], yy.shape = [1000,1000]
    # Merge the features and convert in the usual [Nsample, Nfeatures] shape
    X_fake = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    # Predict the probability
    Z = model.predict_proba(X_fake)
    # And convert values back into a 2D grid for visualization
    Z = Z[:, 1].reshape(xx.shape)
    
    cmap_light = colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
    cmap_solid = colors.ListedColormap(['#0000FF', '#FF0000'])
    # Color the background: pcolormesh colors the background using
    # the grid coordinates and the predicted probability as color
    # eventually filling gaps
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, norm=colors.Normalize(0., 1.))
    # We also plot the contour where the color changes with a grey line
    plt.contour(xx, yy, Z, [0.5], colors='grey')
    # Plot the point
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_solid)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    plt.gca().set_aspect('equal', adjustable='box')
    if legend:
        plt.legend(handles=legend)

    plt.show()
```

```
plot_gaussian(model, model.means_, model.covariance_, X_lda_train, y_lda_train)
```

---

## LDA with Polynomial features

With this analysis, we have two assumptions:  
1. There are no differences in the covariance of all the features
2. The distribution of the features under a specific class can be summarized as a multivariate linear distribution.

```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

model_lda_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('lda', LDA(store_covariance=True))])

params = {"poly__degree":range(1,3)}

cv_lda_poly = GridSearchCV(model_lda_poly, params, refit=True, cv=10, 
                  scoring=make_scorer(accuracy))
cv_lda_poly.fit(X_lda_train, y_lda_train)

```

Obtaining the best degree:

```
cv_lda_poly.best_params_
cv_lda_poly.best_score_
pd.DataFrame(cv_lda_poly.cv_results_)
```

May not be necessary:

```
def show_results(cv, X_lda_test, params, prefix=''):
    prefix = ' '+prefix    
    results = pd.DataFrame(cv.cv_results_)
    # Plot the CV (mean) scores for all possible parameters
    plt.plot(..., ..., label=prefix)

    # Find the best
    best_idx = #...
    # Plot it as a cross
    plt.plot(..., ..., marker='X')
    plt.legend()

    print(prefix, f"(best {results[params][best_idx]}) CV accuracy:",  cv.best_score_)
    print(prefix, f"(best {results[params][best_idx]}) Test accuracy:", accuracy(y_lda_test, cv.best_estimator_.predict(X_lda_test)))
    
show_results(cv, X_lda_test, 'param_poly__degree')
```


























