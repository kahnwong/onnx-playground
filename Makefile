inference-scikit-learn:
	cd onnx-create && uv run onnx_playground/test_inference_scikit_learn.py

inference-onnx:
	cd onnx-create && uv run onnx_playground/test_inference_via_onnx.py
