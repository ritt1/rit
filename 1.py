import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing as f
df = f(as_frame=True).frame
df.hist(figsize=(19,11)); plt.tight_layout(); plt.show()
df.plot.box(subplots=True, layout=(3,3),figsize=(19,11)); plt.tight_layout(); plt.show()
print(f"{c} outliers:", ((df[c]<q1-1.5*(q3-q1))|(df[c]>q3+1.5*(q3-q1))).sum())


