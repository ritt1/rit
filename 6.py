import numpy as np
import matplotlib.pyplot as plt

def lwr(x, X, y, tau):
    w = np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2))
    W = np.diag(w)
    return x @ np.linalg.pinv(X.T@W@X)@X.T@W@y

# Generate data and # Predict
X=np.linspace(0,2*np.pi,100)
y=np.sin(X)+0.1*np.random.randn(100)
X_=np.c_[np.ones(100), X]
T=np.linspace(0,2*np.pi,200) 
T_=np.c_[np.ones(200), T] 
pred=[lwr(t, X_, y,0.5) for t in T_]

# Plot
plt.scatter(X,y,color='red',label='data training')
plt.plot(T,pred,color='blue',label='LWR FIT(tau-0.5)')
plt.legend()
plt.grid()
plt.show()
