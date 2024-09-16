inference-scikit-learn:
	cd onnx-create && uv run onnx_playground/test_inference_scikit_learn.py

inference-onnx-python:
	cd onnx-create && uv run onnx_playground/test_inference_via_onnx.py

inference-onnx-rust:
	cd onnx-serve && cargo run --release
