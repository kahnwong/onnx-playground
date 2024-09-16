import joblib
import numpy as np
from onnx_playground import config

# read model
model = joblib.load(config.SCIKIT_LEARN_MODEL_PATH)

# predict
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
pred = model.predict(data)
print(pred)
