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

# print validation set


X = X_cl_train.to_numpy()
X_t = X_cl_test.to_numpy()

from sklearn.linear_model import LogisticRegression             # 1- model selection
model = LogisticRegression(solver="newton-cg", penalty='none')  # 2- hyperparams
model.fit(X, y_cl_train)  

```
