import onnxruntime as ort
import numpy as np
import os
from loguru import logger

class Decoder:
    def __init__(self, onnxdir: str, nameformat: str, count: int = 32):

        # reload tokenizer
        assert os.path.isdir(onnxdir)

        self.sessions = []
        self.inputs = None
        self.output_names = []

        for idx in range(count):
            filepath = os.path.join(onnxdir, nameformat.format(idx))
            assert os.path.exists(filepath)
            sess = ort.InferenceSession(filepath)
            self.sessions.append(sess)

            if self.inputs is None:
                self.inputs = sess.get_inputs()
                outputs = sess.get_outputs()
                self.output_names = [output.name for output in outputs]

            logger.debug('{} loaded'.format(filepath))

        self.norm = ort.InferenceSession(os.path.join(onnxdir, 'norm.onnx'))
        self.head = ort.InferenceSession(os.path.join(onnxdir, 'head.onnx'))
        logger.debug('head loaded')
        

    def decode(self, _inputs: dict, idx: int):
        assert len(self.inputs) == len(_inputs)
        
        sess = self.sessions[idx]
        output_tensors = sess.run(None, _inputs)
        assert len(output_tensors) == len(self.output_names)

        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor
        
        return output

    def norm_head(self, hidden: np.array):
        outputs = self.norm.run(None, {'input': hidden})
        outputs = self.head.run(None, {'input': outputs[0]})
        return outputs[0]
