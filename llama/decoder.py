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

        from .ort_wrapper import OrtWrapper
        baseline = OrtWrapper("/home/khj/下载/7b-onnx/alpaca-onnx-7B-fp16/models/decoder-merge-{}.onnx".format(idx))
        ort_outputs = baseline.forward(_inputs)

        print('round {}'.format(idx))
        keys = ['hidden_in', 'attn_mask', 'position_ids']
        for key in keys:
            np.save(key, _inputs[key])
        print(np.allclose(outputs['hidden_out'], ort_outputs['hidden_out']))
        print(np.allclose(outputs['past_key'], ort_outputs['past_key']))
        print(np.allclose(outputs['past_value'], ort_outputs['past_value']))
        
        import pdb
        pdb.set_trace()

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
