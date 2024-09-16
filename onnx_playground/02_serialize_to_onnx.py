import os

import joblib
import numpy as np
from skl2onnx import to_onnx
from sklearn.datasets import load_iris

from onnx_playground import config

# read model
model = joblib.load(config.SCIKIT_LEARN_MODEL_NAME)

# prep X, to be used for inferring input type
iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)

# serialize
onx = to_onnx(model=model, X=X[:1])

# save model
os.makedirs(config.ONNX_MODEL_PATH, exist_ok=True)
with open(config.ONNX_MODEL_NAME, "wb") as f:
    f.write(onx.SerializeToString())
