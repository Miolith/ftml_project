import numpy as np

X = np.random.uniform(0.,1., size=10000)
Y = np.array([np.random.exponential(1/(1 + x)) for x in X]).ravel()

def bayes_estimator(x):
    return 1/(1+x)

def ols_estimator(x):
    return (12 - 13 *np.log(2)) * x + 15/2 * np.log(2) - 6

def compute_empirical_risk(f, X, Y):
    predictions = np.array([f(x) for x in X])
    return np.mean((Y - predictions) ** 2)

print("Bayes estimator loss", compute_empirical_risk(bayes_estimator,X,Y))
print("OLS estimator loss", compute_empirical_risk(ols_estimator, X, Y))
