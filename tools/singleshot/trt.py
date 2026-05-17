"""
Create: 2022.11.01
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import sys

sys.path.append(".")

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import os
import numpy as np
import torch

class TRT():
    def __init__(self,
                 args):
        self.trt_logger = trt.Logger()
        self.cfx = cuda.Device(0).make_context()

        if not os.path.isfile(args.trt_path):
            self.engine = self.build_engine(args, self.trt_logger)
            self.context = self.engine.create_execution_context()
        else:
            with open(args.trt_path, "rb") as f:
                plan = f.read()

            with trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(plan)

            self.context = self.engine.create_execution_context()

        self.device_inputs = []
        self.input_shapes = []
        self.device_outputs = []
        self.host_outputs = []
        self.output_shapes = []

        for idx, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                input_shape = self.engine.get_binding_shape(binding)

                self.input_shapes.append(input_shape)

                input_size = trt.volume(input_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize
                device_input = cuda.mem_alloc(input_size)

                self.device_inputs.append(device_input)
            else:
                output_shape = self.engine.get_binding_shape(binding)

                self.output_shapes.append(output_shape)

                host_output = cuda.pagelocked_empty(trt.volume(output_shape) * self.engine.max_batch_size, dtype=np.float32)

                self.host_outputs.append(host_output)

                device_output = cuda.mem_alloc(host_output.nbytes)

                self.device_outputs.append(device_output)

    def build_engine(self,
                     args,
                     trt_logger):
        builder = trt.Builder(trt_logger)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, trt_logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        builder.max_batch_size = args.batch_size

        if getattr(args, "fp16", False) and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        with open(args.onnx_path, "rb") as f:
            print("Begin Onnx file parsing")

            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  [onnx-parser] {parser.get_error(i)}")
                raise RuntimeError(f"failed to parse ONNX file: {args.onnx_path}")

        print(f"Complete Onnx file parsing — inputs={network.num_inputs}, outputs={network.num_outputs}")
        print("Building an engine")

        plan = builder.build_serialized_network(network, config)

        if plan is None:
            raise RuntimeError("TensorRT engine build returned None — check the parser/network logs above")

        os.makedirs(os.path.dirname(args.trt_path) or ".", exist_ok=True)

        with open(args.trt_path, "wb") as f:
            f.write(plan)

        with trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(plan)

        print("Complete creating engine")

        return engine
    
    def inference(self,
                  img,
                  img2lidars):
        self.cfx.push()

        stream = cuda.Stream()

        img = np.array(img.numpy(), dtype=np.float32)
        img2lidars = np.array(img2lidars.numpy(), dtype=np.float32)

        cuda.memcpy_htod_async(self.device_inputs[0], img, stream)
        cuda.memcpy_htod_async(self.device_inputs[1], img2lidars, stream)

        binding_list = []

        for device_input in self.device_inputs:
            binding_list.append(int(device_input))

        for device_output in self.device_outputs:
            binding_list.append(int(device_output))

        self.context.execute_async_v2(bindings=binding_list, stream_handle=stream.handle)
        outputs = []

        for i in range(len(self.device_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.device_outputs[i], stream)
            outputs.append(torch.Tensor(self.host_outputs[i]).cuda().reshape(tuple(self.output_shapes[i])))

        stream.synchronize()
        self.cfx.pop()

        return outputs