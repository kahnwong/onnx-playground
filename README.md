# ONNX Playground

## Safetensors to ONNX

<https://huggingface.co/docs/transformers/en/serialization>

```bash
uv tool install optimum[exporters]
optimum-cli export onnx --model {$MODEL_ID | $MODEL_PATH} data/model-x-onnx/

# example
optimum-cli export onnx --task text-generation --monolith --model /Users/kahnwong/Git/data/llm/THaLLE-0.1-7B-fa data/model-onnx/thalle-0.1-7b/
```

## Refs
- <https://onnx.ai/sklearn-onnx/>
- <https://ort.pyke.io/>
- <https://ort.pyke.io/rustdoc/ort/type.Tensor.html>
