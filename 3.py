import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
iris = load_iris() 
X = iris.data 
y = iris.target 
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X) 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis') 
plt.xlabel('PCA Component 1') 
plt.ylabel('PCA Component 2') 
plt.title('PCA - Iris Dataset (2D)') 
plt.show() 


OR  


from sklearn.datasets import load_iris 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
# Load data 
iris = load_iris() 
X = iris.data 
y = iris.target 
# PCA to 2D 
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X) 
# Plot 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y) 
plt.title("PCA - Iris Dataset") 
plt.show()
