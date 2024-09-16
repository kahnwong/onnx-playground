import numpy as np
import onnxruntime as rt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from onnx_playground import config


# init model
sess = rt.InferenceSession(config.ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

# read data, to be used for inference
iris = load_iris()
X, y = iris.data, iris.target
_, X_test, _, _ = train_test_split(X, y)

# get model input/label name
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
print(pred_onx)
