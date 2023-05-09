import tensorrt as trt
import os
import re
import sys
import argparse
from typing import Any, Dict, Optional, Sequence, Union

def scan(onnxdir, trtdir):

    for idx in range(32):
        fromfile = os.path.join(onnxdir, 'decoder-merge-{}.onnx'.format(idx))
        tofile = os.path.join(trtdir, 'decoder-merge-{}.engine'.format(idx))
        cmd = f'trtexec --onnx={fromfile} --minShapes=hidden_in:1x1x4096,attn_mask:1x1x1x1,position_ids:1x1,past_key_in:1x32x0x128,past_value_in:1x32x0x128  --optShapes=hidden_in:1x1x4096,attn_mask:1x1x1x2,position_ids:1x1,past_key_in:1x32x1x128,past_value_in:1x32x1x128  --maxShapes=hidden_in:1x64x4096,attn_mask:1x1x64x192,position_ids:1x64,past_key_in:1x32x192x128,past_value_in:1x32x192x128  --fp16  --saveEngine={tofile}'
        os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser(description='convert llama.onnx to trt')
    parser.add_argument('onnxdir', help='llama 7B onnx model directory.')
    parser.add_argument('trtdir', help='output trt model directory.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    scan(args.onnxdir, args.trtdir)
