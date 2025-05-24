import numpy as np, matplotlib.pyplot as plt

def lwr(x, X, y, tau):
    w = np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2))
    W = np.diag(w)
    return x @ np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y

X = np.linspace(0, 2*np.pi, 100)[:, None]
y = np.sin(X).ravel() + 0.1*np.random.randn(100)
Xt = np.linspace(0, 2*np.pi, 200)[:, None]
Yt = [lwr(x, X, y, 0.5) for x in Xt]

yp = [lwr(x, Xb, y, tau) for x in Xt_b]

plt.scatter(X, y, c='r', label='Data training ')
plt.plot(Xt, yp, c='b', label='LWR FIT(tau-0.5)')
plt.legend(); plt.grid(); plt.show()

