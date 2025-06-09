from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X, y = load_iris(return_X_y=True)
X_pca = PCA(2).fit_transform(X)
for i, c in enumerate('rgb'): plt.scatter(*X_pca[y==i].T, c=c, label=load_iris().target_names[i])
plt.legend(); plt.title('PCA - Iris'); plt.show()
