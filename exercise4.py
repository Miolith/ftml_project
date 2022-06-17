import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

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
            "RidgeCV": RidgeCV(),
            "LassoCV": LassoCV(),
            "ElasticNetCV": ElasticNetCV(),
            "RandomForestRegressor" : RandomForestRegressor(n_estimators=100, max_depth=15),
            "AdaBoost Regressor" : AdaBoostRegressor(n_estimators=100),
            "Gradient Boosting Regressor" : GradientBoostingRegressor(n_estimators=100)
            }

for (name, model) in model_list.items():
    print(name, "performance :", test_model(model))

clf = GridSearchCV(estimator=Ridge(),
        param_grid={'alpha': range(1, 100)})
clf.fit(X,y)
print("The best parameters for Ridge are :", clf.best_params_)
print("The CV score of this setting is", clf.best_score_)


clf = GridSearchCV(estimator=Lasso(),
        param_grid={'alpha': [.01, .05,.1, .2, .3, .4, .5, .6, .7 ,.8, .9, 1.]})
clf.fit(X,y)
print("The best parameters for Lasso are :", clf.best_params_)
print("The CV score of this setting is", clf.best_score_)


clf = GridSearchCV(estimator=ElasticNet(fit_intercept=True),
        param_grid={'alpha': [0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.]})
clf.fit(X,y)
print("The best parameters for ElasticNet are :", clf.best_params_)
print("The CV score of this setting is", clf.best_score_)
