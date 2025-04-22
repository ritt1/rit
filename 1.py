import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing 
# Load data 
housing = fetch_california_housing(as_frame=True).frame 
# Numerical columns 
numeric_cols = housing.select_dtypes(include=['number']).columns 
# Histograms 
housing[numeric_cols].hist(figsize=(12, 10)) 
plt.suptitle("Histograms") 
plt.tight_layout() 
plt.show() 
# Box plots 
plt.figure(figsize=(12, 10)) 
for i, col in enumerate(numeric_cols): 
plt.subplot(3, 3, i + 1) 
housing.boxplot(column=col) 
plt.title(col) 
plt.suptitle("Box Plots") 
plt.tight_layout() 
plt.show() 
print("Histograms and box plots generated.")
