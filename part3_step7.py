import numpy as np

n = 1000
d = 20


X = np.random.rand(n,d)

theta_star = np.random.randint(1,d, size=d)
y = X @ theta_star

sigma_2 = (1/(n-d) * np.linalg.norm(y - X @ np.linalg.inv(X.T @ X) @ X.T @ y) ** 2)

print("sigma^2 estimation :", sigma_2)
