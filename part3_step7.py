import numpy as np

n = 10000
d = 20


X = np.random.rand(n,d)

theta_star = np.random.randint(1,d, size=d)
y = X @ theta_star

sigma_2 = (1/(n-d) * np.linalg.norm(y - X @ np.linalg.inv(X.T @ X) @ X.T @ y) ** 2)

print("sigma^2 estimation with epsilon=0:   %.2f" % sigma_2)

# Now with epsilon noise

epsilon = np.random.normal(0, 2, size=n)
y = X @ theta_star + epsilon
sigma_2 = (1/(n-d) * np.linalg.norm(y - X @ np.linalg.inv(X.T @ X) @ X.T @ y) ** 2)

print("sigma^2 estimation with epsilon=N(mean:0, sigma:2):   %.2f" % sigma_2)
