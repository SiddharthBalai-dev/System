import numpy as np, pandas as pd
from numpy.linalg import svd

R = np.array([
    [5,3,0,1,0],
    [4,0,0,1,0],
    [1,1,0,5,0],
    [0,0,5,4,0],
    [0,1,5,4,0]
])
users = ['User1','User2','User3','User4','User5']
items = ['Laptop','Phone','Headphones','Camera','Smartwatch']

# Fill missing with user mean
R_filled = np.where(R==0, np.nan, R)
means = np.nanmean(np.where(R==0, np.nan, R), axis=1)
R_filled = np.where(R==0, means, R)


# SVD + reconstruction
U, s, Vt = svd(R_filled)
R_pred = U @ np.diag(s) @ Vt

# Show predicted ratings
print(pd.DataFrame(R_pred, index=users, columns=items))

# Top-N recommendations
for i,u in enumerate(users):
    top = np.argsort(R_pred[i])[::-1][:2]
    print(f"{u}: {[items[j] for j in top]}")
