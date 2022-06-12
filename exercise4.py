import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = np.load("data/regression/inputs.npy")
y = np.load("data/regression/labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X,y)

def test_model(model):
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

print("Datasets proprties :")
print("Nb of samples :", len(y))
print("Nb of features :", len(X[0]))
print("This needs a regression model that can handle a small set of samples and 20 features")
print("Let's try some of them:\n")


model_list = {
            "SVR Linear": SVR(kernel='linear'),
            "SVR rbf": SVR(kernel='rbf'),
            "Ridge alpha=12.0": Ridge(alpha=12.0, random_state=0),
            "Ridge alpha=15.0": Ridge(alpha=15.0, random_state=0),
            "Lasso alpha=12.0": Lasso(alpha=12.0),
            "RandomForestRegressor" : RandomForestRegressor(n_estimators=100, max_depth=15),
            "AdaBoost Regressor" : AdaBoostRegressor(n_estimators=100),
            "Gradient Boosting Regressor" : GradientBoostingRegressor(n_estimators=100)
            }

for (name, model) in model_list.items():
    print(name, "performance :", test_model(model))

print("\nRidge regression seems to be the best regression model in this case")
