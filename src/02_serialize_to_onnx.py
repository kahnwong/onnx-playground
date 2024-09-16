import os

import joblib
import numpy as np
from skl2onnx import to_onnx
from sklearn.datasets import load_iris

# read model
model = joblib.load("data/model/model.joblib")

# prep X, to be used for inferring input type
iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)

# serialize
onx = to_onnx(model=model, X=X[:1])

# save model
os.makedirs("data/model-onnx", exist_ok=True)
with open("data/model-onnx/model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
