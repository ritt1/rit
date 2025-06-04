import seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
X = PCA(2).fit_transform(StandardScaler().fit_transform(load_breast_cancer().data))
y = KMeans(2, random_state=0).fit_predict(X)
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette='Set2'); 
plt.title('K-Means Clustering'); plt.show()
