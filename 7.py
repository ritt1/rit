from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X, y = load_boston(return_X_y=True); X = X[:, [5]]
m=LinearRegression().fit(X,y)
plt.scatter(X, y, s=10,label='Actual');
plt.plot(X, m.predict(X), 'r',label='Predict');
plt.title("Linear Regression")
plt.xlabel("Average Rooms per Dwelling");plt.ylabel("House Price")
plt.legend(); plt.show()


