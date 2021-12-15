import tensorrt as trt
import sys
import argparse


def cli():
    desc = 'Compile Onnx model to TensorRT'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='', help='onnx file location')
    parser.add_argument('-fp', '--floatingpoint', type=int, default=16, help='floating point precision. 16 or 32')
    parser.add_argument('-o', '--output', default='', help='name of trt output file')
    args = parser.parse_args()
    model = args.model 
    fp = args.floatingpoint
    if fp != 16 and fp != 32:
        print('floating point precision must be 16 or 32')
        sys.exit()
    output = 'desiredmodel.trt'
    return {
        'model': model,
        'fp': fp,
        'output': output
    }

"""
takes in onnx model
converts to tensorrt
"""
if __name__ == '__main__':
    args = cli()
    batch_size = 4
    model = '{}'.format(args['model'])
    output = '{}'.format(args['output'])
    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, logger) as parser, trt.Runtime(logger) as runtime:
        # Setting up building configuration
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28
        builder.max_batch_size = batch_size
        if args['fp'] == 16:
            builder.fp16_mode = True

        # Parsing the onnx file 
        with open(model, 'rb') as f:
            print('Beginning ONNX file parsing')
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("ERROR", parser.get_error(error))

        #Setting up input layer 
        print("num layers:", network.num_layers)
        network.get_input(0).shape = [batch_size, 3, 640, 640]  #Change according to the input layer of your model
        print(network.get_input(0).shape)

        #Building deserialized trt engine
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        
        #Serializing the engine and writing it to the output file
        with open(output, 'wb') as f:
            f.write(engine.serialize())
        print("Completed creating Engine")
        