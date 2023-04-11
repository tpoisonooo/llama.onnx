import onnxruntime as ort
import numpy as np
import os
from loguru import logger
from .memory_pool import MemoryPoolSimple

class Decoder:
    
    def __init__(self, pool: MemoryPoolSimple, onnxdir: str, nameformat: str, count: int = 32):

        # reload tokenizer
        assert os.path.isdir(onnxdir)
        self._pool = pool

        for idx in range(count):
            filepath = os.path.join(onnxdir, nameformat.format(idx))
            self._pool.submit('decode{}'.format(idx),filepath)

        self._pool.submit('embed', os.path.join(onnxdir, 'embed.onnx'))
        self._pool.submit('norm', os.path.join(onnxdir, 'norm.onnx'))
        self._pool.submit('head', os.path.join(onnxdir, 'head.onnx'))

    def decode(self, _inputs: dict, idx: int):
        key = 'decode{}'.format(idx)

        handler = self._pool.fetch(key)
        outputs = handler.forward(_inputs)
        
        return outputs

    def embed(self, input_ids: np.array):
        handler = self._pool.fetch('embed')

        input_embed = handler.forward({'input': input_ids})['embed']
        return input_embed

    def norm_head(self, hidden: np.array):
        handler = self._pool.fetch('norm')
        output = handler.forward({'input': hidden})['output']

        handler = self._pool.fetch('head')
        output = handler.forward({'input': output})['output']
        return output
