import numpy as np

X = np.random.uniform(0.,1., size=10000)
Y = np.array([np.random.exponential(1/(1 + x)) for x in X]).ravel()

def bayes_estimator(x):
    return 1/(1+x)

def likelihood_estimator(x):
    l = 1 / np.mean(Y)
    return l * np.exp(-l * x)

def compute_empirical_risk(f, X, Y):
    predictions = np.array([f(x) for x in X])
    return np.mean((Y - predictions) ** 2)

print(compute_empirical_risk(bayes_estimator,X,Y))
print( compute_empirical_risk(likelihood_estimator, X, Y))
