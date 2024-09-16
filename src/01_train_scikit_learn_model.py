import os

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# read data
iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

# save model
os.makedirs("data/model", exist_ok=True)
joblib.dump(clr, "data/model/model.joblib")
