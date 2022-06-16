import numpy as np

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

X = np.load("data/classification/inputs.npy")
y = np.load("data/classification/labels.npy").ravel()

def test_model(model):
    return np.mean(cross_val_score(model, X, y, cv=5))

print("Datasets proprties :")
print("Nb of samples :", len(y))
print("Nb of features :", len(X[0]))
print("This needs a classification model that can handle a small set of samples and 20 features")
print("Let's try some of them:\n")


model_list = {
            "LinearSVC": LinearSVC(max_iter=10000),
            "KNeighbors": KNeighborsClassifier(),
            "SVC" : SVC(max_iter=10000),
            "RandomForestClassifier ": RandomForestClassifier(),
            "AdaBoostClassifier ": AdaBoostClassifier(),
            "GradientBoostingClassifier ": GradientBoostingClassifier()
            }

for (name, model) in model_list.items():
    print(name, "CV score :", test_model(model))

print("\nLinearSVC seem to be the best classification models in this case")

tuned_parameters =  {"C": [1, 10, 100, 1000], "penalty":['l1','l2'], "loss": ["hinge", "squared_hinge"]}


clf = GridSearchCV(estimator=LinearSVC(max_iter=10000),
        param_grid=tuned_parameters)
clf.fit(X,y)

print("The best parameters are :", clf.best_params_)
print("The CV score of this setting is", clf.best_score_)
