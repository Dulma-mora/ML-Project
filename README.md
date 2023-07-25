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

#### Predicting Probabilities and classifying

```
# Predict the CHD probability for being 10 and 70 years old
# Use predict_proba to retrieve the predicted probability
probas = model.predict_proba([[10], [70]])

# Returns P(not CHD|x)=P(0.0|x) and P(CHD|x)=P(1.0|x)
p_chd = probas[:, 1]
print("Probability of AHD for being 10 years old:", p_chd[0])
print("Probability of AHD for being 70 years old:", p_chd[1])
```

#### Statistical tests on the coefficients
```
from scipy.stats import norm, zscore

def z_test(X, y, model, names, alpha=None):
    n_samples, n_features = X.shape
    betas = np.concatenate([model.intercept_, model.coef_.reshape(-1)])
    
    # Compute the prediction
    pred = model.predict_proba(X) # [N, 2]
    y = y.reshape(-1)    
    X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=-1)
    n_samples, n_features = X.shape
    
    V = np.diagflat(np.product(pred, axis=1))
    covLogit = np.linalg.inv(np.dot(np.dot(X.T, V), X))
    se_b = np.sqrt(np.diag(covLogit)) 
    
    z_stat_b = (betas-0)/se_b

    # Compute the p-value (two-sided test)
    p_values = np.array([2 * norm.sf(np.abs(z_stat)) for z_stat in z_stat_b])
    
    df = pd.DataFrame()
    df["Name"] = names
    df["Coefficients"] = betas
    df["Standard Errors"] = np.round(se_b, decimals=4)
    df["Z-stat"] = np.round(z_stat_b, decimals=1)
    df["p-value"] = p_values
    if alpha:
        rejectH0 = p_values < alpha
        df["reject H0"] = rejectH0    
    
    return df

z_test(X, y_train, model, ["Intercept", "Age"], alpha=0.0001)
```

#### Plot the linear regressor boundary
```
# Plot the linear regressor boundary
x_vals = np.linspace(start=0, stop=100, num=100000).reshape(-1,1) # CHECK
y_pred = model.predict(x_vals) # CHECK

# NOTE: predict() in this case returns a binary value (1.0 or 0.0)
y_vals = model.predict_proba(x_vals)
print("y_vals.shape =", y_vals.shape)

# predict_proba returns two values for each data point, providing
# both the P(C0|x) and P(C1|x). We plot the probability of class 1
y_c1 = y_vals[:, 1]

plt.scatter(X_train, y_train)
plt.plot(x_vals, y_pred, color='g')
plt.plot(x_vals, y_c1, color='r')
plt.xlabel("Age")
plt.ylabel("Probability of AHD")

plt.xlim(0,100)
```
















