import onnxruntime as ort
import os
from loguru import logger


class Embed:
    def __init__(self, onnxdir: str):
        # reload tokenizer
        assert os.path.isdir(onnxdir)
        filepath = os.path.join(onnxdir, 'embed.onnx')
        assert os.path.exists(filepath)

        self.sess = ort.InferenceSession(filepath)
        self.inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]
        logger.debug('Embed loaded')

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        output_tensors = self.sess.run(None, _inputs)

        assert len(output_tensors) == len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor
        
        return output
