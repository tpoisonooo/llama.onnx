import numpy as np
from loguru import logger

# refers to https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py 
def warp_topk(tensor: np.array, topk: int, fill_value = -float("Inf")):
    if topk is None or topk <= 0:
        return tensor
    assert len(tensor.shape) == 2
    
    if topk > tensor.shape[-1]:
        logger.warning('topk value {} bigger than tensor shape {}, updated'.format(topk, tensor.shape))
        topk = min(topk, tensor.shape[-1])

    for idx, pval in enumerate(tensor):
        # for each row, loop
        non_topk_idx = np.argpartition(pval, -topk)[0:-topk]
        tensor[idx][non_topk_idx] = fill_value

    return tensor


def warp_temperature(tensor: np.array, temperature: float):
    EPSILON = 1e-4
    if abs(temperature - 1.0) <= EPSILON:
        return tensor
    
    if temperature <= EPSILON:
        raise Exception('bad temperature {}, make sure `0.0 < temperature < 1.0`')
    
    return tensor / temperature
