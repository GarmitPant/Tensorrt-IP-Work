import os
import torch
import torch.nn as nn
import sys
import time
import logging
from pathlib import Path
import math
import argparse
import onnx

ROOT = os.getcwd()
YOLO = os.path.join(ROOT, "yolov5")
ROOT = os.getcwd()
WORK = os.path.join(ROOT, "work")

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['WORK'] = WORK
os.environ['YOLO'] = YOLO

 
def colorstr(*input):
    # Colors a string
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension for yolov5
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def export_onnx(model, im, model_type, opset, train, dynamic, prefix=colorstr('ONNX:')):
    # Function to export Pytorch to ONNX 
    try:

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = model_type.with_suffix('.onnx')

        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        LOGGER.info(f'{prefix} export success, saved as {f}')
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')



def cli():
    desc = 'Convert from Pytorch to ONNX'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='', help='Enter YoloV5s or Resnet50')
    args = parser.parse_args()
    model = args.model
    return {
        'model': model
    }


if __name__ == '__main__':
    
    include = ["onnx"]
    args = cli()
    
    #YOLOv5s/Resnet50 specific. For custom model, change the value according to the training image size.  
    imgsz=(640, 640) 
    
    batch_size = 1


    t = time.time()
    include = [x.lower() for x in include]
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand

    model_type = '{}'.format(args['model'])
    model_type_path = ROOT / model_type
    file = Path(model_type_path)

    
    """
    For user input of pretrained weights, 
    use the following lines of code:
    """
    # weights=ROOT / '<your model>.pt'
    # file = Path(weights)

    
    device = torch.device('cpu')

    # Load PyTorch model
    """
    For demonstration purpose, the models are being 
    loaded from PyTorch Hub:
    """
    if(model_type.lower() == "resnet50"):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    elif(model_type.lower() == "yolov5s"):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        gs = int(model.stride)  # grid size (max stride)
        imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    else:
        print("PLEASE enter valid choice.")
        sys.exit()


    """
    For custom model, initialize the variable <model> with an instance of your model class.
    Then, load the the pre-trained weights using torch.load(PATH) or torch.load_state_dict(toch.load(PATH)).
    You can use the following lines of code:
    """
    # model = YourModelClass(*args, **kwargs)
    # model = torch.load(file)
   
    # Set the model to evaluate mode
    model.eval()
    
    # Input
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  

 

    for _ in range(2):
        y = model(im)  # dry runs
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file}")

    # Exports
    if 'onnx' in include:
        export_onnx(model, im, file, opset=14, train=False, dynamic=False)