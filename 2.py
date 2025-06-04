import seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame
sns.heatmap(df.corr(), annot=True); plt.show()
sns.pairplot(df); plt.show()
