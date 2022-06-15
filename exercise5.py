import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.load("data/classification/inputs.npy")
y = np.load("data/classification/labels.npy")
X_train, X_test, y_train, y_test = train_test_split(X,y)

def test_model(model):
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

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
    print(name, "accuracy :", test_model(model))

print("\nSVC seem to be the best classification models in this case")
