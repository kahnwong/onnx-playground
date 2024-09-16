import numpy as np
import onnxruntime as rt
from onnx_playground import config


# init model
sess = rt.InferenceSession(config.ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

# read data, to be used for inference
# iris = load_iris()
# X, y = iris.data, iris.target
# _, X_test, _, _ = train_test_split(X, y)
data = np.array(
    [
        [5.7, 3.0, 4.2, 1.2],
        [6.3, 3.3, 4.7, 1.6],
        [6.7, 3.1, 5.6, 2.4],
        [5.9, 3.0, 5.1, 1.8],
        [6.7, 3.1, 4.7, 1.5],
        [6.6, 3.0, 4.4, 1.4],
    ]
)

# get model input/label name
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# print(input_name)
# print(label_name)

pred_onx = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
print(pred_onx)
