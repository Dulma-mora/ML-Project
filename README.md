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
print("Test accuracy:", accuracy(y_test, model.predict(X_t)))

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
model.fit(X_full_train, y_train)                               # 3- model fitting
y_predict = model.predict(X_full_train)                        # 4- model testing

print("Train accuracy:", accuracy(y_train, model.predict(X_full_train)))
print("Test accuracy:", accuracy(y_test, model.predict(X_full_test)))

# z test table
z_test(X_full_train, y_train, model, ["Intercept", *X_full_feat], alpha=0.005)
```















