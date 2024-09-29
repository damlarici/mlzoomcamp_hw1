import numpy as np
import pandas as pd
# pd.__version__

df = pd.read_csv("laptops.csv")


df.head()
df.shape

df.columns
df["Brand"].nunique()
df.isnull().sum()

df.groupby("Brand")["Final Price"].max()
df["Screen"].isnull().sum()
df["Screen"].median()
df["Screen"].mode()
df["Screen"].fillna(15.6, inplace=True)

df[df["Brand"] == "Innjoo"]

df[df["Brand"] == "Innjoo"] & df[["RAM", "Storage", "Screen"]]

X = df[df["Brand"] == "Innjoo"][["RAM", "Storage", "Screen"]].to_numpy()

XTX = X.dot(X.T)

XTX_inv = np.linalg.inv(XTX)