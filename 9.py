from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

X, y = fetch_olivetti_faces(shuffle=True, random_state=42, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gnb = GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))

cv = cross_val_score(gnb, X, y, cv=5).mean()
print(f"Cross-val accuracy: {cv*100:.2f}%")
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, img, t, p in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"T:{t} P:{p}")
    ax.axis('off')
plt.show()
