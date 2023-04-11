import numpy as np
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


def npsoftmax(x, axis):
    y = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=axis, keepdims=True)


def npmultinominal2D(x):
    assert len(x.shape) == 2

    ret = np.zeros((x.shape[0], 1), dtype=x.dtype)

    for row, pval in enumerate(x):
        ret[row] = np.random.multinomial(1, pval).argmax()

    return ret


if __name__ == '__main__':
    data = np.ones((12, 8))
    data1 = npsoftmax(data, -1)

    data2 = npmultinominal2D(data1)
    print(data2)
