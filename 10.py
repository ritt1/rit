from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
y_km = KMeans(n_clusters=2, random_state=0).fit_predict(X)
print(confusion_matrix(y, y_km), classification_report(y, y_km))

pca = PCA(2).fit(X)
X2D, cent = pca.transform(X), pca.transform(KMeans(2).fit(X).cluster_centers_)
df = pd.DataFrame(X2D, columns=['PC1','PC2']); df['Cluster'], df['Label'] = y_km, y

for col in ['Cluster','Label']:
    sns.scatterplot(x='PC1', y='PC2', hue=col, data=df).set_title(col); plt.show()

sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df)
plt.scatter(*cent.T, c='k', s=100, marker='X')
plt.title('Centroids'); plt.show()
