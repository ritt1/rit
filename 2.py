import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing 
data = fetch_california_housing() 
df = pd.DataFrame(data.data, columns=data.feature_names) 
sns.heatmap(df.corr(), annot=True, cmap='coolwarm') 
plt.title("Correlation Heatmap") 
plt.show() 
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]) 
plt.show() 
