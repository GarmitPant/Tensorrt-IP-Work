# TensorRT-IP-Work
Pipelines to convert PyTorch and TensorFlow models to ONNX and further convert ONNX models to TensorRT engines.
The pipelines are made to be used on linux systems.

## How to use

### 1) PyTorch to ONNX conversion
For converting models trained in PyTorch with saved weights in .pt or .pth format, use the file `pt2onnx.py`.
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


### 2) Tensorflow to ONNX conversion (Linux only)
For converting models trained in Tensorflow and saved in saved_model format, use the notebook `tfsavedmodel2onnx.ipynb`.
Follow the instruction in the comments to convert custom models or run the provided examples. Install tf2onnx python library for this file using the following commands:
```bash
$ pip install --user -U tf2onnx
```


### 3) ONNX to TensorRT conversion
For converting models saved in .onnx format, use the file `onnx2trt.py`.
Use the following commands to convert:
```bash
$ python3 onnx2trt.py -e ONNXFILEPATH
```
Use the following commands to get help regarding the file:
```bash
$ python3 onnx2trt.py --help
```

### Installing and Setting up TensorRT Python API
Follow this link to install and setup TensorRT 8.x.x.x on your system:
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing


## Requirements
<ul>
  <li>Python >= 3.7</li>
  <li>TensorRT >= 8.0.x.x</li>
  <li>CUDA >= 11.x</li>
  <li>cuDNN == 8.2.1</li>
  <li>PyCUDA</li>
  <li>tf2onnx</li>
  <li>matplotlib >= 3.2.2</li>
  <li>numpy >= 1.18.5</li>
  <li>Pillow >= 7.1.2</li>
  <li>scipy >= 1.4.1</li>
  <li>torch >= 1.7.0</li>
  <li>torchvision >= 0.8.1</li>
  <li>onnx >= 1.9.0 </li>
</ul>


## References
 <ul>
  <li>https://docs.nvidia.com/deeplearning/tensorrt/</li>
  <li>https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/</li>
  <li>https://github.com/ultralytics/yolov5</li>
  <li>https://github.com/wang-xinyu/tensorrtx</li>
  <li>https://github.com/onnx/tensorflow-onnx</li>
</ul>











