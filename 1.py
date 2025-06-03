import seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing as fch
df = fch(as_frame=True).frame
for c in df.select_dtypes('number'):
    sns.histplot(df[c]); plt.title(c); plt.show()
    sns.boxplot(x=df[c]); plt.title(c); plt.show()
    q1, q3 = df[c].quantile([.25, .75])
    print(f'{c}: {(df[c] < q1 - 1.5*(q3-q1)).sum() + (df[c] > q3 + 1.5*(q3-q1)).sum()} outliers')
print(df.describe())
