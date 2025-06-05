from sklearn.datasets import fetch_olivetti_faces 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 

X,y=fetch_olivetti_faces(return_X_y=True) 
Xt,Xs,yt,ys=train_test_split(X,y,test_size=0.3) 
p=GaussianNB().fit(Xt,yt).predict(Xs) 

print("Acc:",accuracy_score(ys,p)) 
plt.imshow(Xs[0].reshape(64,64),cmap='gray'); 
plt.title(f"T:{ys[0]} P:{p[0]}");

plt.axis('off'); 
plt.show() 
