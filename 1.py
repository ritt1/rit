import seaborn as sns; from sklearn.datasets import fetch_california_housing as f
df = f(as_frame=True).frame
df.hist(figsize=(8,5)); plt.tight_layout(); plt.show()
df.plot.box(subplots=True, layout=(3,3),figsize=(8,5)); plt.tight_layout(); plt.show()
for c in df: q1,q3=df[c].quantile([.25,.75]);
print(f"{c} outliers:", ((df[c]<q1-1.5*(q3-q1))|(df[c]>q3+1.5*(q3-q1))).sum())


