# llama.onnx

## News

04/?? deploy <llama.onnx, quant table> to aarch64

04/18 export mixed-precision quant table from [GPTQ-for-llama](https://github.com/qwopqwop200/GPTQ-for-LLaMa/pull/189)

04/11 add 13GB onnx-fp16 models

04/11 add memory pool, support 2GB RAM laptop :star:

04/10 reduce onnx model size to 26GB

04/10 support `temperature` add `topk` logits warp

04/07 add [onnxruntime demo](demo-single.py)

04/05 init project


## Features

* Release llama 7B onnx models
* With a 400-lines onnxruntime alpaca demo
  * neither `torch` nor `transformers` required
  * support memory pool, works on 2GB laptop/PC (very slow :turtle:)

Why do this ?
1. **Visualization**. `graphviz` crashed on llama model. LLM visualization tool must support nest or operator folding feature
2. **Quatization**. LLM often repeat itself, just like [fractal](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal_small.gif). For llama quantization, loading part of decoder backbone would be enough (400MB). It could be quantized partially
3. **Embeded device**. Small board IO error occurs when `dd` a big single file
4. **Distributed system**. Inference LLM on many hybrid (FPGA/NPU/GPGPU) devices would be simple
5. **onnx tools**. Device manufacturer has support onnx well, there is no reason to neglect it

## Usage

Download onnx models here:

| Precision | Size | URL |
| :-: | :-: | :-: |
| fp32 | 26GB | [huggingface](https://huggingface.co/tpoisonooo/alpaca.onnx/tree/main) |
| fp16 | 13GB | [huggingface](https://huggingface.co/tpoisonooo/alpaca.onnx/tree/fp16) or [硬件模型库](https://platform.openmmlab.com/deploee/onnx-list) |

Here is the graph to call them:

![](./images/onnx-flow.jpg)

Try `onnxruntime` demo, no `torch` required, and the precision has been checked.

```bash
$ python3 -m pip install -r requirements.txt
$ python3 demo-single.py ${FP16_ONNX_DIR} "bonjour"
..
# If you only have 4GB memory, use `--poolsize`
$ python3 demo-single.py ${FP16_ONNX_DIR} "bonjour" --poolsize 4
..
Bonjour.

# Try more options
$ python3 demo-single.py --help
```


## Export onnx

**STEP1 Convert to HF format**

These models converted from [alpaca huggingface](https://github.com/tatsu-lab/stanford_alpaca).

- If you are using [LLaMa](https://github.com/facebookresearch/llama) or [llama.cpp](https://github.com/ggerganov/llama.cpp), convert it to HF format first. Here are steps:
    ```bash
    # install transformers master
    $ git clone https://github.com/huggingface/transformers
    $ cd transformers && python3 setup.py install
    ..
    $ cd src/transformers
    $ python3 src/transformers/models/llama/convert_llama_weights_to_hf.py  --input_dir ${LLaMa_PATH}  --model_size 7B  --output_dir ${HF_PATH}
    ```

- If you are using [alpaca-lora](https://github.com/tloen/alpaca-lora), use [this script](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_llama_with_chinese_lora_to_hf.py) to merge LoRA weights.

- If you are using [alpaca](https://github.com/tatsu-lab/stanford_alpaca), go STEP2.

**STEP2 `torch.onnx.export`**

Checkout transformers to this [hacking branch](https://github.com/tpoisonooo/transformers/tree/add-convert), run single inference.

```bash
$ python3 tools/export-onnx.py ${PATH_ALPACA_7B}
```

**STEP3 convert to fp16**

Use `onnxconverter-common.float16`

```bash
$ cd tools
$ python3 -m pip install -r requirements.txt
$ python3 convert-fp32-to-fp16.py ${FP32_PATH} ${FP16_PATH}
```

## Notes
1. Any `logits_processor` or `BeamSearch` not implemented, so the result would be not good
2. I have compared the output values of `onnxruntime-cpu` and `torch-cuda`, and the maximum error is 0.002, not bad
3. The current state is equivalent to these configurations
```bash
temperature=0.1
total_tokens=2000
top_p=1.0
top_k=40
repetition_penalty=1.0
```


## Acknowlegements
* [llama](https://github.com/facebookresearch/llama)
* [alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* [alpaca-lora](https://github.com/tloen/alpaca-lora)
* [transformers](https://github.com/huggingface/transformers)
* [peft](https://github.com/huggingface/peft)
* [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)


## License
[GPLv3](why-gpl.md)
