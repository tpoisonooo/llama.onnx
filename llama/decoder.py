import onnxruntime as ort
import numpy as np
import os
from loguru import logger
from threading import Lock


def singleton(cls):
    _instance = {}
    _instance_lock = Lock()

    def inner(*args, **kwargs):
        if cls not in _instance:
            with _instance_lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


@singleton
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


@singleton
class Norm:

    def __init__(self, onnxdir: str):
        # reload tokenizer
        assert os.path.isdir(onnxdir)
        filepath = os.path.join(onnxdir, 'norm.onnx')
        assert os.path.exists(filepath)

        self.sess = ort.InferenceSession(filepath)
        self.inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]
        logger.debug('Norm loaded')

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        output_tensors = self.sess.run(None, _inputs)

        assert len(output_tensors) == len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor

        return output


@singleton
class Head:

    def __init__(self, onnxdir: str):
        # reload tokenizer
        assert os.path.isdir(onnxdir)
        filepath = os.path.join(onnxdir, 'head.onnx')
        assert os.path.exists(filepath)

        self.sess = ort.InferenceSession(filepath)
        self.inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]
        logger.debug('Head loaded')

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        output_tensors = self.sess.run(None, _inputs)

        assert len(output_tensors) == len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor

        return output


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

        self._embed = Embed(onnxdir)
        self._norm = Norm(onnxdir)
        self._head = Head(onnxdir)

    def decode(self, _inputs: dict, idx: int):
        assert len(self.inputs) == len(_inputs)

        sess = self.sessions[idx]
        output_tensors = sess.run(None, _inputs)
        assert len(output_tensors) == len(self.output_names)

        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor

        return output

    def embed(self, input_ids: np.array):
        input_embed = self._embed.forward({'input': input_ids})['embed']
        return input_embed

    def norm_head(self, hidden: np.array):
        output = self._norm.forward({'input': hidden})['output']
        output = self._head.forward({'input': output})['output']
        return output
