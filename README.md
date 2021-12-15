# TensorRT-IP-Work
Pipelines to convert PyTorch and TensorFlow models to ONNX and further convert ONNX models to TensorRT engines.

## How to use

### 1) PyTorch to ONNX conversion
For converting models trained in PyTorch with saved weights in .pt or .pth format, use the file pt2onnx.py.
Use the following commands to get:
1) YoloV5s.onnx:
```bash
$ python3 pt2onnx.py -m YoloV5s
```
2) Resnet50.onnx
```bash
$ python3 pt2onnx.py -m Resnet50
```

For custom models, you can follow the instructions given in the source code comments to make the proposed changes.


### 2) Tensorflow to ONNX conversion
For converting models trained in Tensorflow and saved in saved_model format, use the notebook tfsavedmodel2onnx.ipynb.
Follow the instruction in the comments to convert custom models or run the provided examples.


### 3) ONNX to TensorRT conversion
For converting models saved in .onnx format, use the file onnx2trt.py.
Use the following commands to convert:
```bash
$ python3 onnx2trt.py -m Resnet50
```
