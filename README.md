# llama.onnx

Features

* release llama 7B onnx models
* and a 400-lines python script to run it
* neither `torch` nor `transformers` required

So you can quantize llama partially and optimize kernel step by step. 

## Usage
Please download it here
* huggingface https://huggingface.co/tpoisonooo/llama.onnx/tree/main
* BaiduYun https://pan.baidu.com/s/195axYNz79U6YkJLETNJmXw?pwd=onnx

Here is the graph to call them:

![](./images/onnx-flow.jpg)

Try `onnxruntime` demo, no `torch` required, and the precision has been checked.

```bash
$ python3 -m pip install -r requirements.txt
$ python3 demo-single.py ${ONNX_DIR} "bonjour"
..
Bonjour.
```

## Updates

2023/04/?? reduce onnx model size

2023/04/10 support `temperature` add `topk` logits warp

2023/04/07 add [onnxruntime demo](demo-single.py) and `tokenizer.model` (do not forget to download it)

2023/04/05 init project


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
* [alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* [llama](https://github.com/facebookresearch/llama)
* [transformers](https://github.com/huggingface/transformers)
* [peft](https://github.com/huggingface/peft)
* [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)


## License
[GPLv3](why-gpl.md)
