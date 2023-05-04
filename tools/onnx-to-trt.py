import tensorrt as trt
import os
import re
import sys
import argparse
from typing import Any, Dict, Optional, Sequence, Union

from packaging import version

def save(engine: Any, path: str) -> None:
    """Serialize TensorRT engine to disk.

    Args:
        engine (Any): TensorRT engine to be serialized.
        path (str): The absolute disk path to write the engine.
    """
    with open(path, mode='wb') as f:
        if isinstance(engine, trt.ICudaEngine):
            engine = engine.serialize()
        f.write(bytearray(engine))


def search_cuda_version() -> str:
    """try cmd to get cuda version, then try `torch.cuda`
    Returns:
        str: cuda version, for example 10.2
    """

    version = None

    pattern = re.compile(r'[0-9]+\.[0-9]+')
    platform = sys.platform.lower()

    def cmd_result(txt: str):
        cmd = os.popen(txt)
        return cmd.read().rstrip().lstrip()

    if platform == 'linux' or platform == 'darwin' or platform == 'freebsd':  # noqa E501
        version = cmd_result(
            " nvcc --version | grep  release | awk '{print $5}' | awk -F , '{print $1}' "  # noqa E501
        )
        if version is None or pattern.match(version) is None:
            version = cmd_result(
                " nvidia-smi  | grep CUDA | awk '{print $9}' ")

    elif platform == 'win32' or platform == 'cygwin':
        # nvcc_release = "Cuda compilation tools, release 10.2, V10.2.89"
        nvcc_release = cmd_result(' nvcc --version | find "release" ')
        if nvcc_release is not None:
            result = pattern.findall(nvcc_release)
            if len(result) > 0:
                version = result[0]

        if version is None or pattern.match(version) is None:
            # nvidia_smi = "| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |" # noqa E501
            nvidia_smi = cmd_result(' nvidia-smi | find "CUDA Version" ')
            result = pattern.findall(nvidia_smi)
            if len(result) > 2:
                version = result[2]

    if version is None or pattern.match(version) is None:
        try:
            import torch
            version = torch.version.cuda
        except Exception:
            pass

    return version


def from_onnx(onnx_model: str,
              output_filepath: str,
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              int8_param: Optional[dict] = None,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              **kwargs) -> trt.ICudaEngine:
    """Create a tensorrt engine from ONNX.

    Args:
        onnx_model (str or onnx.ModelProto): Input onnx model to convert from.
        output_file_prefix (str): The path to save the output ncnn file.
        input_shapes (Dict[str, Sequence[int]]): The min/opt/max shape of
            each input.
        max_workspace_size (int): To set max workspace size of TensorRT engine.
            some tactics and layers need large workspace. Defaults to `0`.
        fp16_mode (bool): Specifying whether to enable fp16 mode.
            Defaults to `False`.
        int8_param (dict): A dict of parameter  int8 mode. Defaults to `None`.
        device_id (int): Choice the device to create engine. Defaults to `0`.
        log_level (trt.Logger.Severity): The log level of TensorRT. Defaults to
            `trt.Logger.ERROR`.

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine created from onnx_model.

    Example:
        >>> from mmdeploy.apis.tensorrt import from_onnx
        >>> engine = from_onnx(
        >>>             "onnx_model.onnx",
        >>>             {'input': {"min_shape" : [1, 3, 160, 160],
        >>>                        "opt_shape" : [1, 3, 320, 320],
        >>>                        "max_shape" : [1, 3, 640, 640]}},
        >>>             log_level=trt.Logger.WARNING,
        >>>             fp16_mode=True,
        >>>             max_workspace_size=1 << 30,
        >>>             device_id=0)
        >>>             })
    """

    if device_id != 0:
        import os
        old_cuda_device = os.environ.get('CUDA_DEVICE', None)
        os.environ['CUDA_DEVICE'] = str(device_id)
        if old_cuda_device is not None:
            os.environ['CUDA_DEVICE'] = old_cuda_device
        else:
            os.environ.pop('CUDA_DEVICE')

    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)

    # TODO: use TorchAllocator as builder.gpu_allocator

    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        parse_valid = parser.parse_from_file(onnx_model)
    else:
        raise TypeError('Unsupported onnx model type!')

    if not parse_valid:
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()

    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     max_workspace_size)
    else:
        config.max_workspace_size = max_workspace_size

    cuda_version = search_cuda_version()
    if cuda_version is not None:
        version_major = int(cuda_version.split('.')[0])
        if version_major < 11:
    # cu11 support cublasLt, so cudnn heuristic tactic should disable CUBLAS_LT # noqa E501
            tactic_source = config.get_tactic_sources() - (
                1 << int(trt.TacticSource.CUBLAS_LT))
            config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    if config.add_optimization_profile(profile) < 0:
        logger.warning(f'Invalid optimization profile {profile}.')

    if fp16_mode:
        if not getattr(builder, 'platform_has_fast_fp16', True):
            logger.warning('Platform does not has fast native fp16.')
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    # create engine
    if hasattr(builder, 'build_serialized_network'):
        engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'

    save(engine, output_filepath)
    return engine

# onnx_model: str,
#               output_filepath: str,
#               input_shapes: Dict[str, Sequence[int]],
#               max_workspace_size: int = 0,
#               fp16_mode: bool = False,
#               int8_mode: bool = False,
#               int8_param: Optional[dict] = None,
#               device_id: int = 0,
#               log_level: trt.Logger.Severity = trt.Logger.ERROR,
def scan(onnxdir, trtdir):
    MAX_INPUT_LEN = 64
    MAX_TEXT_LEN = 192
    if not os.path.exists(trtdir):
        os.mkdir(trtdir)
    
    input_shapes = {
        'hidden_in': {'min_shape': [1,1,4096], 'opt_shape': [1,1,4096], 'max_shape': [1,MAX_INPUT_LEN,4096]},
        'attn_mask': {'min_shape': [1,1,1,1], 'opt_shape': [1,1,1,2], 'max_shape': [1,1, MAX_INPUT_LEN,MAX_TEXT_LEN]},
        'position_ids': {'min_shape': [1,1], 'opt_shape': [1,1], 'max_shape': [1,MAX_INPUT_LEN]},
        'past_key_in': {'min_shape': [1,32,0,128], 'opt_shape': [1,32,1,128], 'max_shape': [1,32,MAX_TEXT_LEN,128]},
        'past_value_in': {'min_shape': [1,32,0,128], 'opt_shape': [1,32,1,128], 'max_shape': [1,32,MAX_TEXT_LEN,128]}
    }

    for home, _, files in os.walk(onnxdir):
        for filename in files:
            filepath = os.path.join(home, filename)
            if 'decoder' in filename:
                from_onnx(onnx_model=filepath, output_filepath='decoder.engine', fp16_mode=True, input_shapes=input_shapes, max_workspace_size=5 << 30)

def parse_args():
    parser = argparse.ArgumentParser(description='convert llama.onnx to trt')
    parser.add_argument('onnxdir', help='llama 7B onnx model directory.')
    parser.add_argument('trtdir', help='output trt model directory.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    scan(args.onnxdir, args.trtdir)
