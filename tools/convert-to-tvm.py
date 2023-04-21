import argparse
import onnx
from tvm import relay
import tvm
import os
import sys


def convert(filepath: str, outdir: str):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    basename = os.path.basename(filepath)

    onnx_model = onnx.load(filepath)
    mod, params = relay.frontend.from_onnx(onnx_model,
                                           dtype='float32',
                                           opset=14)

    _exec = relay.vm.compile(mod, target='llvm', params=params)

    bytecode, lib = _exec.save()

    with open(os.path.join(outdir, "{}.bytecode".format(basename)),
              mode='wb') as f:
        f.write(bytecode)

    lib.export_library(os.path.join(outdir, "{}.so".format(basename)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert onnx model to tvm.vm format')
    parser.add_argument('onnxpath', help='onnx file path.')
    parser.add_argument('outdir', help='output tvm.vm directory.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    convert(args.onnxpath, args.outdir)
