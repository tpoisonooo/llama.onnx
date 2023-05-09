import tensorrt as trt
import os
from loguru import logger
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtWrapper:
    def __init__(self, enginefile: str):
        assert os.path.exists(enginefile)

        self.enginefile = enginefile

        logger = trt.Logger(trt.ILogger.WARNING)
        with open(enginefile, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        """Load input/output names from engine."""
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names

        output_names = list(set(names) - set(input_names))
        self._output_names = output_names

    def forward(self, _inputs: dict):
        stream = cuda.Stream()

        mem_holders = []
        bindings = []

        # import pdb
        # pdb.set_trace()

        N = 0
        lastN = 0
        sumN = 0

        # prepare input
        for k,v in _inputs.items():
            if k == 'hidden_in':
                N = v.shape[1]
            elif k == 'attn_mask':
                sumN = v.shape[-1]
            elif k == 'past_key_in':
                lastN = v.shape[-2]

            nbytes = v.nbytes
            if nbytes > 0:
                device_mem = cuda.mem_alloc(nbytes)
                cuda.memcpy_htod_async(device_mem, v, stream)
            else:
                device_mem = cuda.mem_alloc(4)
            mem_holders.append(device_mem)
            bindings.append(int(device_mem))
            self.context.set_binding_shape(self.engine.get_binding_index(name=k), tuple(v.shape))
        
        # prepare output
        outputs = []
        output_shapes = {}
        for k in self._output_names:
            shape = self.engine.get_binding_shape(name=k)
            if k == 'hidden_out':
                shape[1] = N
            elif k == 'past_key' or k == 'past_value':
                shape[2] = sumN
            output_shapes[k] = shape
            
            size = trt.volume(shape)
            dtype = self.engine.get_binding_dtype(name=k)
            dtype_map = {
                trt.DataType.HALF: np.float16,
                trt.DataType.FLOAT: np.float32,
                trt.DataType.INT32: np.int32,
                trt.DataType.INT8: np.int8
            }
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype_map[dtype])
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            outputs.append(HostDeviceMem(host_mem, device_mem))

        # Run inference.
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()

        ret = {}
        for idx, k in enumerate(self._output_names):
            shape = output_shapes[k]
            ret[k] = outputs[idx].host.reshape(shape)
        return ret


    def __del__(self):
        logger.debug('{} unload'.format(self.enginefile))


import onnxruntime as ort
import os
from loguru import logger

class OrtWrapper:
    def __init__(self, onnxfile: str):
        assert os.path.exists(onnxfile)
        self.onnxfile = onnxfile
        self.sess = ort.InferenceSession(onnxfile)
        self.inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]
        logger.debug('{} loaded'.format(onnxfile))

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        output_tensors = self.sess.run(None, _inputs)

        assert len(output_tensors) == len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor

        return output
    
    def __del__(self):
        logger.debug('{} unload'.format(self.onnxfile))



if __name__ == '__main__':
    # prepare input
    keys = ['hidden_in', 'attn_mask', 'position_ids']
    _inputs = {}
    for key in keys:
        _inputs[key] = np.load(open('../data/{}.npy'.format(key), 'rb'))
    
    _inputs['past_key_in'] = np.zeros((1, 32, 0, 128), dtype=np.float16)
    _inputs['past_value_in'] = np.zeros((1, 32, 0, 128), dtype=np.float16)

    # inference with trt
    trt_wrapper = TrtWrapper('/home/khj/下载/7b-onnx/alpaca-onnx-7B-fp16/trt/decoder-merge-0.engine')
    trt_outputs = trt_wrapper.forward(_inputs)

    # with ort
    ort_wrapper = OrtWrapper('/home/khj/下载/7b-onnx/alpaca-onnx-7B-fp16/models/decoder-merge-0.onnx')
    ort_outputs = ort_wrapper.forward(_inputs)

    # compare
    print(np.allclose(trt_outputs['hidden_out'], ort_outputs['hidden_out']))
    print(np.allclose(trt_outputs['past_key'], ort_outputs['past_key']))
    print(np.allclose(trt_outputs['past_value'], ort_outputs['past_value']))
    
    diff1 = trt_outputs['hidden_out'] - ort_outputs['hidden_out']
    print(diff1.max())