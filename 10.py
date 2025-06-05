import seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
X = PCA(2).fit_transform(StandardScaler().fit_transform(load_breast_cancer().data))
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y); 
plt.title('K-Means Clustering');
plt.xlabel('PC1');plt.ylabel('PC2');plt.show()

