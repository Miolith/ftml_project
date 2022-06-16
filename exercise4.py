import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


X = np.load("data/regression/inputs.npy")
y = np.load("data/regression/labels.npy")


def test_model(model):
    return np.mean(cross_val_score(model, X, y.ravel(), cv=5))

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

clf = GridSearchCV(estimator=Ridge(),
        param_grid={'alpha': range(1, 100)})
clf.fit(X,y)
print("The best parameters are :", clf.best_params_)
print("The CV score of this setting is", clf.best_score_)
