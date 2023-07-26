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

## Quadratic Discriminant Analysis
```
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

model = QDA(...) # do we need to write something here?
model = model.fit(X_lda_train, y_lda_train)

print("Train accuracy:", accuracy(y_lda_train, model.predict(X_lda_train)))
print("Test accuracy:", accuracy(y_lda_test, model.predict(X_lda_test)))

plot_gaussian(model, model.means_, np.stack(model.covariance_, axis=0), X_lda_train, y_lda_train, 2) #this changes
```

## KNN Analysis

```
from sklearn.model_selection import GridSearchCV

model_knn = KNeighborsClassifier()
params_knn = {'n_neighbors': range(1, 15)}
folds=10
scorer = #...

cv_knn = GridSearchCV(model_knn, params_knn, refit=True, cv=10,
                     scoring=make_scorer(accuracy))
cv_knn.fit(X_cl_train, y_cl_train)
```

```
show_results(cv_knn, X_cl_test, "param_n_neighbors")
```

#### Normalizating values

```
from sklearn.preprocessing import StandardScaler, MinMaxScaler

params = {'knn__n_neighbors': range(1, 20)}
model_std = Pipeline([
    ('norm', ...),
    ('knn', KNeighborsClassifier())])
cv_std = GridSearchCV(model_std, params, refit=True, cv=10,
                     scoring=make_scorer(accuracy))
cv_std.fit(X_cl_train, y_cl_train)

model_minmax = Pipeline([
    ('norm', ...),
    ('knn', KNeighborsClassifier())])
cv_minmax = GridSearchCV(model_minmax, params, refit=True, cv=10,
                        scoring=make_scorer(accuracy))
cv_minmax.fit(X_cl_train, y_cl_train)

# Plot the cv lines and the comparison between non-normalized and normalized values
show_results(cv_knn, X_cl_test, "param_n_neighbors", prefix="Unnormalized")
show_results(cv_std, X_cl_test, 'param_knn__n_neighbors', "StandardScaler")
show_results(cv_minmax, X_cl_test, 'param_knn__n_neighbors', "MinMaxScaler")
```

#### Comparing all models

```
lda = LDA()
lda_poly = Pipeline([('poly', PolynomialFeatures(degree=2)),
                     ('lda', LDA(store_covariance=True))])
qda = QDA()
knn = KNeighborsClassifier(n_neighbors=11)
knn_std = Pipeline([
    ('norm', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=16))])
knn_minmax = Pipeline([
    ('norm', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=17))])

from collections import OrderedDict
models = OrderedDict([('lda', lda), ('lda_poly', lda_poly), ('qda', qda),
                      ('knn', knn), ('knn_std', knn_std), ('knn_minmax', knn_minmax)])

cv_scores, test_scores = [], []
for name, model in models.items():
    scores = #...
    cv_score = #...
    cv_scores.append(cv_score)
    
    model.fit(X_full_train, y_train)
    test_score = accuracy(y_test, model.predict(X_full_test))
    test_scores.append(test_score)
    print("{} CV score: {:.4f},  test score {:.4f}".format(name, cv_score, test_score))

data = pd.DataFrame()
data['model'] = list(models.keys()) * 2
data['metric'] = ['10-cv accuracy'] * len(cv_scores) + ['test accuracy'] * len(test_scores)
data['score'] = cv_scores + test_scores

sns.barplot(x='model', y='score', data=data, hue='metric')
plt.legend(loc='lower right')
```
---

## Support Vector Machine SVM Analysis

This step should start with a linearly separable case, e.g., **classifying just two classes (setosa vs rest) in the example dataset** using the sepal characteristics. **We expect the SVM classifier to learn a robust boundary line, wrt the training data.**     
In this case, we should also separate a categorical feature (?) so this code its needed to be finished:

```
# using the same parameters of previous steps
# lda variables can also apply
sns.scatterplot(X_cl_train[:,0], X_cl_train[:,1], hue=y_cl_train, marker='o', label="train") 
sns.scatterplot(X_cl_test[:,0], X_cl_test[:,1], hue=y_cl_test, marker='^', label="test")
```


```
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# We set a very high C value, i.e., virtually
# disabling regularization
model_svm = SVC(kernel='linear', C=np.inf)
model_svm.fit(X_cl_train, y_cl_train)

train_acc = accuracy_score(model_svm.predict(X_cl_train), y_cl_train)
test_acc = accuracy_score(model_svm.predict(X_cl_test), y_cl_test)

print("SVM train accuracy:", train_acc)
print("SVM test accuracy:", test_acc)
```

Here we are verifying the value of `decision_function` (the SVM scores) in correspondence of the support vectors

```
model_svm.support_vectors_
model_svm.decision_function(model_svm.support_vectors_)
```

Creating function to see the borderline decision:

```
def plot_svm_line(model, Xrange, Yrange, label=None):
    Xmin, Xmax = Xrange
    Ymin, Ymax = Yrange
    # Create grid to evaluate model
    xx = np.linspace(Xmin, Xmax, 100)
    yy = np.linspace(Ymin, Ymax, 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.reshape(-1), YY.reshape(-1)]).T # [nsamples, 2]
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    c = plt.contour(XX, YY, Z, colors='g', 
                # We want to plot lines where the decision function is either -1, 0, or 1
                levels=[-1, 0, 1],
                # We set different line styles for each "decision line"
                linestyles=['--','-','--'])
    c.collections[1].set_label(label)
    # Remove this to add +1/-1/0 labels
    # plt.clabel(c, inline=1, fontsize=10)
    # plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k')
```

Using the function. This may not be completely neccesary and need to be modified to fit out current data:

```
X0_range = (X_cl_train[:,0].min(), X_cl_train[:,0].max()) # TODO:
X1_range = (X_cl_train[:,1].min(), X_cl_train[:,1].max()) # TODO:

sns.scatterplot(X_cl_train[:,0], X_cl_train[:,1], hue=y_cl_train, marker='o')
sns.scatterplot(X_cl_test[:,0], X_cl_test[:,1], hue=y_cl_test, marker='^')
plot_svm_line(model_vsm, X0_range, X1_range)
```

#### Non linearly separable cases

Now that we trained our algorithm with linear separable data, we can do something with our non-linearly separable cases.     
First we need to change de C value to a smaller number. Again, this code need to be modified after understanding which features we are supposed to use!

```
# Let's set the C parameter to smaller values
model_svm = SVC(kernel='linear', C=10)
model_svm.fit(X_cl_train, y_cl_train)

sns.scatterplot(X_cl_train[:,0], X_cl_train[:,1], hue=y_cl_train, marker='o') # TODO:
sns.scatterplot(X_cl_test[:,0], X_cl_test[:,1], hue=y_cl_test, marker='^') # TODO:
plot_svm_line(model_svm, X0_range, X1_range)
```

> Some remarks of the notes:      
> By doing so, we enable points to be inside the margin. **The smaller the C, the larger the margin and the larger the number of support vectors.** Notice that slack variables are required for the optimization problem to have a solution if the problem is not linearly separable.            
> Note that even if the margin is larger, it is actually still defined to have a fixed length of $\quad2 = \frac{2}{||\mathbf{w}||}$ wrt the values produced with the model, i.e., the decision function. All the points whose predicted value smaller than $1$ or larger than $-1$ are support vectors by definition. 

```
# Check the value of the decision function 
# in correspondence with support vectors
model_svm.decision_function(model_smv.support_vectors_)
```

In summary: SVM's intention is to find the most robust solution, this is, a **decision line**. The decision line that we are looking for is the one that is the **most far away from the training points possible!**.  

## Comparing SVM with Logistic Regressor & Perceptron

Now, we can compare the border decision of SVM with other linear algorithms that also compute linear bounds (Logistic regressor & Perceptron).    
Note: here we are changing some variable names. Also, we can still choose if we want to use cl or lda parameters. I decided to continue with cl to not confuse myself. In fact, I can also delete lda objects and continue with my life 💀😆

```
from sklearn.linear_model import LogisticRegression, Perceptron

# Train the SVM
svm_model = SVC(kernel='linear', C=100) 
svm_model.fit(X_cl_train, y_cl_train)

# Train a LogisticRegressor
lr_model = LogisticRegression(fit_intercept=True)
lr_model.fit(X_cl_train, y_cl_train)

# Train a Perceptron classifier (try verbose=True)
pt_model = Perceptron()
pt_model.fit(X_cl_train, y_cl_train)

for m in [svm_model, lr_model, pt_model]:
    train_acc = accuracy_score(y_cl_train, m.predict(X_cl_train))
    test_acc = accuracy_score(y_cl_test, m.predict(X_cl_test))
    print("{} train score: {}".format(m.__class__.__name__, train_acc))
    print("{} test score: {}".format(m.__class__.__name__, test_acc))
```

Using a function that should work perfectly cause I did not move anything ☠️

```
# Shows the learned linear models
# intercept_ + coef_[0]*x + coef_[1]*y = 0
def plot_linear_line(model, Xrange, label=None):
    Xmin, Xmax = Xrange
    w0, = model.intercept_
    w1, w2 = model.coef_.flatten()

    x1, y1 = Xmin, -(w0 + w1 * Xmin) / w2
    x2, y2 = Xmax, -(w0 + w1 * Xmax) / w2
    sns.lineplot([x1, x2], [y1, y2], label=label)
```
#### Separable Linear Case

THEREFORE, this code should perfectly work! (after setting the correct parameters). We should visualize the linear boundaries decided per each algorithm. Also, we expect something like this :suspect: :   
- The SVM boundary is the one most far away from the two different classes
- Logistic Regression maximizes the likelihood of points, therefore having a similar effect to the SVM
- The Perceptron is the "less robust" of the three methods. The training algorithm is indeed iterative, and it stops as soon as all the training values are correctly classified. In the last iteration (depending on the learning rate) it may move the line just enough to correctly classify the last remaining misclassified sample, resulting in a line very close to some training data

```
f = sns.scatterplot(X_cl_train[:,0], X_cl_train[:,1], hue=y_cl_train, marker='o') # TODO:
f = sns.scatterplot(X_cl_test[:,0], X_cl_test[:,1], hue=y_cl_test, marker='^') # TODO:

plot_svm_line(svm_model, X0_range, X1_range, label="SVM")
plot_linear_line(pt_model, X0_range, label="Perceptron")
plot_linear_line(lr_model, X0_range, label="LogisticRegression")
_ = plt.axis([X0_range[0]-0.5, X0_range[1]+0.5, X1_range[0]-0.5, X1_range[1]+0.5])
```

#### Non-linearly separable problem

Ok, so we just did it with a linear-separable case, let's compare the three algorithms with a non-linearly separable problem!!!























