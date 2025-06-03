from sklearn.datasets import load_breast_cancer 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
import matplotlib.pyplot as plt 
X, y = load_breast_cancer(return_X_y=True) 
model = DecisionTreeClassifier().fit(X, y) 
pred = model.predict([X[0]]) 
print("Class:", "Benign" if pred[0] == 1 else "Malignant") 
plot_tree(model, filled=True) 
plt.title("Decision tree-breast cancer dataset") 
plt.show()
