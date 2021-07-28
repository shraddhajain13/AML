import numpy as np
from sklearn.linear_model import LinearRegression
#X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3

y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
print(reg.coef_)
pre = reg.predict(X)
print(y)
print(pre)
R = y - pre
print(R)
#reg.coef_
#array([1., 2.])
#reg.intercept_

#reg.predict(np.array([[3, 5]]))
#array([16.])
#a = np.array([1, 2])
#print(a.shape)