import os
import shutil
import onnx
import onnxsim
import argparse
from loguru import logger
from onnxconverter_common import float16
import pdb


def main(_input: str, output: str):
    if not os.path.exists(output):
        os.mkdir(output)

    for home, _, files in os.walk(_input):
        for basename in files:

            if not basename.endswith('.onnx'):
                continue

            logger.debug('simplify and convert {}'.format(basename))
            fp32_model = onnx.load(os.path.join(home, basename))

            inferred_model, check = onnxsim.simplify(fp32_model)
            assert check == True

            fp16_model = float16.convert_float_to_float16(inferred_model)
            onnx.save(fp16_model, os.path.join(output, basename))
        break


def parse_args():
    parser = argparse.ArgumentParser(description='llama.onnx onnxruntime demo')
    parser.add_argument('onnxdir', help='llama 7B onnx model directory.')
    parser.add_argument('outdir', help='output fp16 directory.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.onnxdir, args.outdir)
